#!/usr/bin/env python
# -*- coding: utf-8 -*-

__version__ = "0.1.0"

import logging
from enum import Enum
from logging import LoggerAdapter, basicConfig
from os import environ

FORMAT = "[%(levelname)8s | %(filename)s:%(lineno)s - %(module)s.%(funcName)s() ] %(message)s"
basicConfig(format=FORMAT, level=environ.get("LOSSY_MPI_LOG", "INFO").upper())


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


class Singleton(type):
    """
    class MySingleton(metaclass=Singleton):
        ...
    Setting `metaclass=Singleton` in the classes meta descriptor marks it as a
    singleton object: if the object has already been constructed elsewhere in
    the code, subsequent calls to the constructor just return this original
    instance.
    """

    # Stores instances in a dictionary:
    # {class: instance}
    _instances = dict()

    def __call__(cls, *args, **kwargs):
        """
        Metclass __call__ operator is called before the class constructor -- so
        this operator will check if an instance already exists in
        Singleton._instances. If it doesn't call the constructor and add the
        instance to Singleton._instances. If it does, then don't call the
        constructor and return the instance instead.
        """
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(
                *args, **kwargs
            )

        return cls._instances[cls]


class PoolCount(metaclass=Singleton):
    def __init__(self):
        """
        Singleton counter for Pool transactions
        """
        self.__ct = 0

    def __call__(self):
        val = self.__ct
        self.__ct += 1
        return val

    @property
    def val(self):
        return self.__ct

    def advance(self, ct):
        self.__ct += ct

