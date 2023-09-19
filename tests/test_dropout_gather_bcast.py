#!/usr/bin/env python
# -*- coding: utf-8 -*-

from random import randint
from sys import argv

from lossy_mpi.pool import Pool, Status
from mpi4py import MPI

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
data = None

while True:
    # simulate graceful failure: if no more work, drop rank from pool
    # decide on current status (i.e. if current rank has data to send)
    if n_data <= 0:
        pool.drop()

    # update mask -- this is not necessary, but it does speeding things up
    pool.sync_mask()

    # decide to break (root checks if all done) + root: print mask
    if rank == root:
        if verbose:
            print(pool.mask, flush=True)

        if pool.done:
            break
    else:
        # if done, break out of loop
        if pool.status is Status.DONE:
            break

    # make one datum and sent to root
    if n_data > 0:
        data = randint(101, 200)
        n_data -= 1

    # communicate data
    all_data = pool.gather(data)
    # construct payload for bcast: the sum to all values on all ranks
    if rank == root:
        sum_data = sum([d for d in all_data if d is not None])
    else:
        sum_data = None
    # after gather, bcast the sum to all other ranks
    sum_data = pool.bcast(sum_data)
    if rank == 0 or rank == 2:
        print(f"{rank=} {sum_data=}")

    # print data
    if rank == root:
        print(all_data, sum_data)

last_n_data = comm.gather(n_data, root=root)
last_data = comm.gather(data, root=root)

if rank == root:
    print(f"{last_n_data=}")
    print(f"{last_data=}")

comm.barrier()
