#!/usr/bin/env python
# -*- coding: utf-8 -*-

from logging import getLogger
from enum    import auto, unique

from .       import AutoEnum
from .comms  import TimeoutComm, OperatorMode


LOGGER = getLogger(__name__)


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

        self._root    = root
        self._is_root = self.rank == root

        self._mask = [Status.UNINIT for i in range(self.size)]

        LOGGER.debug(f"Initialized pool at {root=}")

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

    def _exec_transaction(self, sendbuf, recvbuf, failover, mode):
        """
        Gather data from masked ranks -- excluding "dead ranks". If a timemout
        occurs, assign the `failover` value.
        """
        LOGGER.debug(f"Entering transacton, using {mode=}")
        init_op, fini_op = OperatorMode.get(mode, self.comm)
        reqs = list()

        # initiate communications ----------------------------------------------
        if self.is_root:
            LOGGER.debug(f"Root is initializing communications on {self.rank=}")
            # Initiate comms with all ranks
            for i in range(self.size):
                # don't do anything for the root, except updating the data array
                if i == self.root:
                    recvbuf[i] == sendbuf
                    continue
                # don't receive mask data from ranks that are set to "DONE"
                if Status.is_dead(self.mask[i]):
                    LOGGER.debug(f"Source {i=} is considered DEAD, skipping")
                    continue
                # receive mask
                reqs.append((i, init_op(source=i, tag=1)))
                LOGGER.debug(f"Added source {i=} to requests")
        else:
            # make sure that the channel is clear
            if self.last_req_completed and (self._last_req is not None):
                LOGGER.debug(
                    f"Waiting for last isend to complete on {self.rank=}"
                )
                self.last_req.wait()
            # send data
            LOGGER.debug(f"Initiating isend on {self.rank=}")
            self.last_req = fini_op(sendbuf, dest=self.root, tag=1)

        # complete communications ----------------------------------------------
        if self.is_root:
            # Collect requests with timeout
            LOGGER.debug(f"Collecting requests on {self.rank=}")
            self.safe_req_wait(recvbuf, failover, reqs, 1)

    def Gather(self, sendbuf, recvbuf, failover=None):
        """
        Gather data from masked ranks -- excluding "dead ranks". If a timemout
        occurs, assign the `failover` value. Executed in UPPER mode
        """
        LOGGER.debug(f"Start Gather on {self.rank=}")
        self._exec_transaction(
            sendbuf, recvbuf, failover, OperatorMode.UPPER
        )
 
    def gather(self, data, failover=None):
        """
        Gather data from masked ranks -- excluding "dead ranks". If a timemout
        occurs, assign the `failover` value. Executed in UPPER mode
        """
        LOGGER.debug(f"Start Gather on {self.rank=}")
        recvbuf = [failover for i in range(self.size)]
        self._exec_transaction(
            data, recvbuf, failover, OperatorMode.LOWER
        )
        return recvbuf

    def sync_mask(self):
        """
        Syncs masks accross all ranks -- excluding "dead ranks"
        """
        LOGGER.debug(f"Start mask synk on {self.rank=}")
        self._exec_transaction(
            self.status, self.mask, Status.TIMEOUT, OperatorMode.LOWER
        )

    @property
    def done(self):
        """
        Drop the current rank for the pool
        """
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

