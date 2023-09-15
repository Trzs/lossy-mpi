#!/usr/bin/env python
# -*- coding: utf-8 -*-

from enum import Enum, auto, unique

from .comms import TimeoutComm


class AutoEnum(Enum):
    def _generate_next_value_(name, start, count, last_values):
        return count


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

    def gather(self, data_in, data_out, failover):
        """
        Gather data from masked ranks -- excluding "dead ranks". If a timemout
        occurs, assign the `failover` value.
        """
        reqs = list()

        # initiate communications ----------------------------------------------
        if self.is_root:
            # Initiate comms with all ranks
            for i in range(self.size):
                # don't do anything for the root, except updating the data array
                if i == self.root:
                    data_out[i] == data_in
                    continue
                # don't receive mask data from ranks that are set to "DONE"
                if Status.is_dead(self.mask[i]):
                    continue
                # receive mask
                reqs.append(
                    (i, self.comm.irecv(source=i, tag=1))
                )
        else:
            # make sure that the channel is clear
            if self.last_req_completed and (self._last_req is not None):
                self._last_req.wait()
            # send mask
            self._last_req = self.comm.isend(data_in, dest=self.root, tag=1)

        # complete communications ----------------------------------------------
        if self.is_root:
            # Collect requests with timeout
            self.rec(data_out, failover, reqs, 1)

    def comm_mask(self):
        """
        Syncs masks accross all ranks -- excluding "dead ranks"
        """
        self.gather(self.status, self.mask, Status.TIMEOUT)

    def comm_data(self, data):
        """
        Send data from this rank to root -- comms only take place over active
        ranks
        """
        all_data = [None for i in range(self.size)]

        if self.is_root:
            for i in range(self.size):
                # don't do anything for the root
                if i == self.root:
                    all_data[i] = data
                    continue

                # don't do anything for nodes that are not "ready"
                if self.mask[i] is not Status.READY:
                    continue

                # receive data
                req = self.comm.irecv(source=i, tag=2)
                all_data[i] = req.wait()
        else:
            # send data
            req = self.comm.isend(data, dest=self.root, tag=2)
            req.wait()

        return all_data


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

