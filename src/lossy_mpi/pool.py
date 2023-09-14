#!/usr/bin/env python
# -*- coding: utf-8 -*-

from mpi4py import MPI
from enum   import Enum, auto, unique
from time   import sleep 


class AutoEnum(Enum):

    def _generate_next_value_(name, start, count, last_values):
        return count


@unique
class Status(AutoEnum):
    READY = auto()
    DONE = auto()
    UNINIT = auto()
    TIMEOUT = auto()


def is_dead(rank):
    if rank is Status.DONE:
        return True

    if rank is Status.TIMEOUT:
        return True

    return False


class Pool(object):
    def __init__(self, comm, root, timeout, n_tries):
        # Start everything in an uninitialized state
        self._status = Status.UNINIT

        # Assumption: com, rank, size, and root do not change
        self._comm = comm
        self._size = comm.Get_size()
        self._rank = comm.Get_rank()

        self._root    = root
        self._is_root = self.rank == root

        self._timeout = timeout
        self._n_tries = n_tries

        self._last_req = None
 
        self._mask = [Status.UNINIT for i in range(self.size)]

    @property
    def status(self):
        return self._status

    @property
    def comm(self):
        return self._comm

    @property
    def size(self):
        return self._size

    @property
    def rank(self):
        return self._rank

    @property
    def root(self):
        return self._root

    @property
    def timeout(self):
        return self._timeout

    @property
    def n_tries(self):
        return self._n_tries

    @property
    def is_root(self):
        return self._is_root

    @property
    def mask(self):
        return self._mask

    @property
    def last_req_completed(self):
        """
        Check if the last request has been completed
        """
        if self._last_req is None:
            return True
        return self._last_req.test()

    def ready(self):
        self._status = Status.READY

    def drop(self):
        self._status = Status.DONE

    def timeout_rec(self, data, failover, reqs, tag):
        """
        Collect data from reqs with tag -- if timed out, place $failover in its
        place
        """
        for i, req in reqs:
            print(f"{i=}, {req=}")
            # Default to failover
            data[i] = failover

            # try n_tries many times to get a response, if none is received in
            # $timeout seconds, the failover value is not overwritten
            while self.n_tries:
                if self.comm.iprobe(source=i, tag=tag):
                    data[i] = req.wait()
                    success = True
                    break
                else:
                    sleep(self.timeout/self.n_tries)
            
    def comm_mask(self):
        """
        Syncs masks accross all ranks -- excluding "dead ranks"
        """
        reqs = list()

        # initiate communications
        if self.is_root:
            # Initiate comms with all ranks
            for i in range(self.size):
                # don't do anything for the root
                if i == self.root:
                    self.mask[i] == self.status
                    continue

                # don't receive mask data from ranks that are set to "DONE"
                if is_dead(self.mask[i]):
                    continue

                # receive mask
                reqs.append(
                    (i, self.comm.irecv(source=i, tag=1))
                )
        else:
            # send mask -- but only if the channel is clear
            if self.last_req_completed:
                self._last_req = self.comm.isend(
                    self.status, dest=self.root, tag=1
                )

        # complete communications
        if self.is_root:
            # Collect requests with timeout
            self.timeout_rec(self.mask, Status.TIMEOUT, reqs, 1)


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

