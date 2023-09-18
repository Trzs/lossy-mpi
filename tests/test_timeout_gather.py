#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sys import argv

from mpi4py         import MPI
from lossy_mpi.pool import Pool, Status

from random import randint


verbose = False
if len(argv) > 1:
    if argv[1].strip() == "verbose":
        verbose = True


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
root = 0


pool = Pool(comm, root, timeout=2, n_tries=10)
pool.ready()

n_data = randint(1, 10)
data   = None

while True:
    # simulate unexpected failure: if no more work, then stop responding
    # decide on current status (i.e. if current rank has data to send)
    if n_data <= 0:
        break

    # update mask -- this is not necessary, but it does speeding things up
    pool.sync_mask()

    # decide to break (root checks if all done) + root: print mask
    if rank == root:
        if verbose: print(pool.mask, flush=True)

        if pool.done:
            break
    else:
        # if done, break out of loop
        if pool.status is Status.DONE:
            break

    # make one datum and sent to root
    if n_data > 0:
        data    = randint(101, 200)
        n_data -= 1

    # communicate data
    all_data = pool.gather(data)

    # print data
    if rank == root:
        print(all_data)

last_n_data = comm.gather(n_data, root=root)
last_data   = comm.gather(data, root=root)

if rank == root:
    print(f"{last_n_data=}")
    print(f"{last_data=}")

comm.barrier()

