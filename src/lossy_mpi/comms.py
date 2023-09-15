#!/usr/bin/env python
# -*- coding: utf-8 -*-

from time import sleep 


class TimeoutComm(object):
    def __init__(self, comm, timeout, n_tries):
        # Assumption: com, rank, size, and root do not change
        self._comm = comm
        self._size = comm.Get_size()
        self._rank = comm.Get_rank()

        self._timeout = timeout
        self._n_tries = n_tries

        self._last_req = None

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
    def last_req_completed(self):
        """
        Check if the last request has been completed
        """
        if self._last_req is None:
            return True
        flag, _ = self._last_req.test()
        return flag

    def rec(self, data, failover, reqs, tag):
        """
        Collect data from reqs with tag -- if timed out, place $failover in its
        place
        """
        for i, req in reqs:
            # Default to failover
            data[i] = failover
            # try n_tries many times to get a response, if none is received in
            # $timeout seconds, the failover value is not overwritten
            for _ in range(self.n_tries):
                flag, message = req.test()
                if flag:
                    data[i] = message
                    break
                else:
                    sleep(self.timeout/self.n_tries)

