#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest


@pytest.mark.mpi(min_size=2)
def test_mpi():
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    assert comm.size >= 2
