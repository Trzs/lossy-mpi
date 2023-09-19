#!/usr/bin/env python
# -*- coding: utf-8 -*-

from enum import auto, unique

from . import AutoEnum, getLogger
from .comms import OperatorMode, TimeoutComm

LOGGER = getLogger(__name__)


@unique
class Signal(AutoEnum):
    OK = auto()
    TIMEOUT = auto()


@unique
class Status(AutoEnum):
    READY = auto()
    DONE = auto()
    UNINIT = auto()
    TIMEOUT = auto()

    @classmethod
    def is_dead(cls, rank):
        """
        Return True if status is DONE or TIMEOUT
        """
        if rank is cls.DONE:
            return True

        if rank is cls.TIMEOUT:
            return True

        return False


class Pool(TimeoutComm):
    def __init__(self, comm, root, timeout, n_tries):
        # Start everything in an uninitialized state
        self._status = Status.UNINIT

        # Assumption: com, rank, size, and root do not change
        super().__init__(comm, timeout, n_tries)

        self._root = root
        self._is_root = self.rank == root

        self._mask = [Status.UNINIT for i in range(self.size)]

        LOGGER.debug(f"Initialized pool at {root=}", comm=self)

    @property
    def status(self):
        return self._status

    @property
    def root(self):
        return self._root

    @property
    def is_root(self):
        return self._is_root

    @property
    def mask(self):
        return self._mask

    def ready(self):
        self._status = Status.READY

    def drop(self):
        self._status = Status.DONE

    def _exec_gather_transaction(self, sendbuf, recvbuf, failover, mode):
        """
        Gather data from masked ranks -- excluding "dead ranks". If a timemout
        occurs, assign the `failover` value.
        """
        LOGGER.debug(f"Entering gather transacton, using {mode=}", comm=self)
        recv_op, send_op = OperatorMode.get(mode, self.comm)

        # initiate communications ----------------------------------------------
        if self.is_root:
            LOGGER.debug("Root is initializing communications", comm=self)
            # Initiate comms with all ranks
            for i in range(self.size):
                # don't do anything for the root, except updating the data array
                if i == self.root:
                    recvbuf[i] = sendbuf
                    continue
                # don't receive mask data from ranks that are set to "DONE"
                if Status.is_dead(self.mask[i]):
                    LOGGER.debug(f"Source {i=} is considered DEAD, skipping", comm=self)
                    continue
                # receive mask
                LOGGER.debug("Initiating recv", comm=self)
                self.push_req(i, recv_op(source=i, tag=1))
        else:
            # send data
            LOGGER.debug("Initiating send", comm=self)
            self.push_req(0, send_op(sendbuf, dest=self.root, tag=1))

        # complete communications ----------------------------------------------
        # Collect requests with timeout
        self.safe_collect_deferred_req(failover)
        # Assigned collected data to recvbuf
        if self.is_root:
            LOGGER.debug("Collecting requests", comm=self)
            for i, msg in self.deferred_msg.items():
                recvbuf[i] = msg

    def _exec_bcast_transaction(self, sendbuf, recvbuf, failover, mode):
        """
        Scatter data to masked ranks -- excluding "dead ranks". If a timemout
        occurs, assign the `failover` value.
        """
        LOGGER.debug(f"Entering scatter transacton, using {mode=}", comm=self)
        recv_op, send_op = OperatorMode.get(mode, self.comm)

        # index of result in recvbuf
        recvbuf_result_idx = 0

        # initiate communications ----------------------------------------------
        if self.is_root:
            LOGGER.debug("Root is initializing communications", comm=self)
            # Initiate comms with all ranks
            for i in range(self.size):
                # don't do anything for the root, except updating the data array
                if i == self.root:
                    recvbuf[recvbuf_result_idx] = sendbuf
                    continue
                # don't receive mask data from ranks that are set to "DONE"
                if Status.is_dead(self.mask[i]):
                    LOGGER.debug(f"Source {i=} is considered DEAD, skipping", comm=self)
                    continue
                # receive mask
                LOGGER.debug("Initiating recv", comm=self)
                self.push_req(i, send_op(sendbuf, dest=i, tag=2))
        else:
            # send data
            LOGGER.debug("Initiating send", comm=self)
            self.push_req(recvbuf_result_idx, recv_op(source=self.root, tag=2))

        # complete communications ----------------------------------------------
        # Collect requests with timeout
        self.safe_collect_deferred_req(failover)
        # Assigned collected data to recvbuf
        if not self.is_root:
            LOGGER.debug("Collecting requests", comm=self)
            recvbuf[recvbuf_result_idx] = self.deferred_msg[recvbuf_result_idx]

    def Gather(self, sendbuf, recvbuf, failover=None):
        """
        Gather data from masked ranks -- excluding "dead ranks". If a timemout
        occurs, assign the `failover` value. Executed in UPPER mode
        """
        LOGGER.debug("Start Gather", comm=self)
        self._exec_gather_transaction(sendbuf, recvbuf, failover, OperatorMode.UPPER)

    def gather(self, data, failover=None):
        """
        Gather data from masked ranks -- excluding "dead ranks". If a timemout
        occurs, assign the `failover` value. Executed in LOWER mode
        """
        LOGGER.debug("Start gather", comm=self)
        recvbuf = [failover for i in range(self.size)]
        self._exec_gather_transaction(data, recvbuf, failover, OperatorMode.LOWER)
        return recvbuf

    def Bcast(self, buf, failover=None):
        """
        Bcast data accross masked ranks -- excluding "dead ranks", If a timeout
        occurs, assign the `failover` value. Excecuted in UPPER mode
        """
        LOGGER.debug("Start Barrier", comm=self)
        self._exec_bcast_transaction(buf, buf, failover, OperatorMode.UPPER)

    def bcast(self, obj, failover=None):
        """
        Bcast data accross masked ranks -- excluding "dead ranks", If a timeout
        occurs, assign the `failover` value. Excecuted in LOWER mode
        """
        LOGGER.debug("Start barrier", comm=self)
        recvbuf = [failover]
        self._exec_bcast_transaction(obj, recvbuf, failover, OperatorMode.LOWER)
        return recvbuf[0]

    def Barrier(self):
        """
        Barrier on all masked ranks -- exlcuding "dead ranks". Non-dead ranks
        can still time out. If that occurs, the barrier proceeds.
        """
        LOGGER.debug("Start Barrier", comm=self)
        sendbuf = [Signal.OK for i in range(self.size)]
        recvbuf = [None for i in range(self.size)]
        self._exec_gather_transaction(
            sendbuf, recvbuf, Signal.TIMEOUT, OperatorMode.LOWER
        )
        if Signal.TIMEOUT in recvbuf:
            LOGGER.info("Receiving unexpected timeouts", comm=self)

    def barrier(self):
        """
        Barrier on all masked ranks -- exlcuding "dead ranks". Non-dead ranks
        can still time out. If that occurs, the barrier proceeds. Same as
        uppercase "Barrier"
        """
        LOGGER.debug("Start barrier", comm=self)
        self.Barrier()

    def sync_mask(self):
        """
        Syncs masks accross all ranks -- excluding "dead ranks"
        """
        LOGGER.debug("Start sync'ing masks", comm=self)
        self._exec_gather_transaction(
            self.status, self.mask, Status.TIMEOUT, OperatorMode.LOWER
        )

    @property
    def done(self):
        """
        Drop the current rank for the pool
        """
        LOGGER.debug("Dropping this rank from pool", comm=self)
        if not self.is_root:
            return False

        break_root = True
        for i, status in enumerate(self.mask):
            if i == self.root:
                continue

            if status is Status.READY:
                break_root = False
                break

        return break_root
