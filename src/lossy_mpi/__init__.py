#!/usr/bin/env python
# -*- coding: utf-8 -*-

__version__ = "0.1.0"

import logging

from os      import environ
from logging import basicConfig, LoggerAdapter
from enum    import Enum


FORMAT = "[%(levelname)8s | %(filename)s:%(lineno)s - %(module)s.%(funcName)s() ] %(message)s"
basicConfig(
    format = FORMAT,
    level  = environ.get("LOSSY_MPI_LOG", "INFO").upper()
)


class MPIStyleAdapter(LoggerAdapter):
    # def __init__(self, logger, extra=None):
    #     super().__init__(logger, extra)

    def process(self, msg, kwargs):
        comm = kwargs.pop("comm", None)
        if comm is not None:
            msg = f"{comm.rank=} > " + msg
        return msg, kwargs


def getLogger(name):
    return MPIStyleAdapter(logging.getLogger(name), None)


class AutoEnum(Enum):
    def _generate_next_value_(name, start, count, last_values):
        return count

