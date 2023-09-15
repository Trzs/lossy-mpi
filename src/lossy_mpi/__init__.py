#!/usr/bin/env python
# -*- coding: utf-8 -*-

__version__ = "0.0.1"

from os      import environ
from logging import basicConfig
from enum    import Enum

FORMAT = "[%(levelname)8s | %(filename)s:%(lineno)s - %(module)s.%(funcName)s() ] %(message)s"
basicConfig(
    format = FORMAT,
    level  = environ.get("LOSSY_MPI_LOG", "INFO").upper()
)


class AutoEnum(Enum):
    def _generate_next_value_(name, start, count, last_values):
        return count


