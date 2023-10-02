#!/usr/bin/env python
# -*- coding: utf-8 -*-

from enum import auto, unique
from time import sleep
from mpi4py import MPI

from . import AutoEnum, getLogger

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

        # used by deferred requests: requests are a list of (key, val) tuples,
        # messages are a {key: value} dict
        self._deferred_req = list()
        self._rejected_req = list()
        self._deferred_msg = dict()

        LOGGER.debug(f"Initialized Timeout Communicator with {timeout=} and {n_tries=}")

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
    def deferred_req(self):
        """
        List of deferred MPI requests together with each request's index. After
        collection, responses (messages) are stored in `deferred_msg[index]`.
        """
        return self._deferred_req

    @property
    def deferred_msg(self):
        """
        Dictionary (key=idx) of deferred MPI messages
        """
        return self._deferred_msg

    def push_req(self, idx, req):
        """
        Add MPI request to `deferred_req`. Messages -- once collected -- will be
        stored in `deferred_msg[idx]`.
        """
        LOGGER.debug(f"Appending request to index {idx=}", comm=self)
        self._deferred_req.append((idx, req))

    def safe_collect_deferred_req(self, failover, tag):
        """
        Collect (with timeout) all deferred requests, and then delete that list.
        Messages are collected with a timeout. If a request times out, $failover
        is stored in its place.
        """
        LOGGER.debug("Collecting deferred requests", comm=self)
        self._deferred_msg = dict()
        self.safe_req_wait(self._deferred_msg, failover, self._deferred_req, tag)
        self._deferred_req = list()
        # "Rescue" requests that don't have matching tags
        for r in self._rejected_req:
            LOGGER.debug(f"Rejected: {r=}", comm=self)
            self._deferred_req.append(r)
        self._rejected_req = list()

    def safe_req_wait(self, data, failover, reqs, tag):
        """
        Collect data from reqs -- if timed out, place $failover in its place
        """
        LOGGER.debug("Entering safe wait", comm=self)

        for i, req in reqs:
            # Default to failover
            data[i] = failover
            # try n_tries many times to get a response, if none is received in
            # $timeout seconds, the failover value is not overwritten
            try_ct = 0
            while True:
                status = MPI.Status()
                flag, message = req.test(status)
                LOGGER.debug(f"Looking for message {i=}: {flag=} {tag=}", comm=self)
                if flag and (status.Get_tag() == tag):
                    LOGGER.debug(
                        f"Tag match for: {flag=} {status.tag=}, {tag=}", comm=self
                    )
                    data[i] = message
                    break
                elif flag and (status.Get_tag() != tag):
                    # Ignore send req's
                    if status.count == 0:
                        break
                    LOGGER.debug(
                        f"Tag mismatch for: {flag=} {status.tag=}, {tag=}", comm=self
                    )
                    if (i, req) not in self._rejected_req:
                        LOGGER.info(f"{req=}")
                        self._rejected_req.append((i, req))
                else:
                    try_ct += 1
                    if try_ct > self.n_tries:
                        break
                    LOGGER.debug(f"Sleeping for message {i=}", comm=self)
                    sleep(self.timeout / self.n_tries)
