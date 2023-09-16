#!/usr/bin/env python
# -*- coding: utf-8 -*-

from time    import sleep 
from enum    import auto, unique

from .       import getLogger, AutoEnum


LOGGER = getLogger(__name__)


@unique
class OperatorMode(AutoEnum):
    UPPER = auto()
    LOWER = auto()

    @classmethod
    def get(cls, op, comm):
        """
        Returns the mpi4py function corresponding to the operator $op
        """
        if op == cls.UPPER:
            return comm.Irecv, comm.Isend
        if op == cls.LOWER:
            return comm.irecv, comm.isend

        raise RuntimeError(f"Invalid Mode {op=}")


class TimeoutComm(object):
    def __init__(self, comm, timeout, n_tries):
        # Assumption: com, rank, size, and root do not change
        self._comm = comm
        self._size = comm.Get_size()
        self._rank = comm.Get_rank()

        self._timeout = timeout
        self._n_tries = n_tries

        self._last_req = None
        self._last_req_message = None

        # used by deferred requests: requests are a list of (key, val) tuples,
        # messages are a {key: vaule} dict
        self._deferred_req = list()
        self._deferred_msg = dict()

        LOGGER.debug(
            f"Initialized Timeout Communicator with {timeout=} and {n_tries=}"
        )

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
    def timeout(self):
        return self._timeout

    @property
    def n_tries(self):
        return self._n_tries

    @property
    def last_req(self):
        return self._last_req

    @last_req.setter
    def last_req(self, value):
        self._last_req = value

    @property
    def deferred_req(self):
        return self._deferred_req

    @property
    def deferred_msg(self):
        return self._deferred_msg

    def push_req(self, idx, req):
        LOGGER.debug(f"Appending request to index {i=}", comm=self)
        self._deferred_req.append((i, req))

    def safe_collect_deferred_req(self, failover):
        self._deferred_msg = dict()
        self.safe_req_wait(self._deferred_msg, failover)
        self._deferred_req = list()

    @property
    def last_req_completed(self):
        """
        Check if the last request has been completed
        """
        LOGGER.debug(f"Checking {self._last_req=}", comm=self)

        if self._last_req is None:
            return True

        flag, self._last_req_message = self._last_req.test()

        LOGGER.debug(f"Returning {flag=}", comm=self)
        return flag

    def safe_req_wait(self, data, failover, reqs, tag):
        """
        Collect data from reqs with tag -- if timed out, place $failover in its
        place
        """
        LOGGER.debug("Entering safe wait", comm=self)

        for i, req in reqs:
            # Default to failover
            data[i] = failover
            # try n_tries many times to get a response, if none is received in
            # $timeout seconds, the failover value is not overwritten
            for _ in range(self.n_tries):
                flag, message = req.test()
                LOGGER.debug(f"Looking for message {i=}: {flag=}", comm=self)
                if flag:
                    data[i] = message
                    break
                else:
                    LOGGER.debug(f"Sleeping for message {i=}", comm=self)
                    sleep(self.timeout/self.n_tries)
