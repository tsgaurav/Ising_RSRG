#!/usr/bin/env python3
from mpi4py import MPI
import numpy as np
import fasteners, time, threading


comm = MPI.COMM_WORLD
rank = comm.Get_rank() # get your process ID


if rank == 0: # The master is the only process that reads the file
    data = [[1]]*comm.size
else:
    data = None

# Divide the data among processes
data = comm.scatter(data, root=0)

result = []
for item in data:
    i = 0
    while i<3:
        lock = fasteners.InterProcessLock('/tmp/tmplockfile')
        gotten = lock.acquire(blocking=True)  # wait until the lock is available
        if gotten:
            try:
                with open('temp.txt', 'a') as file:
                    file.write(str(rank)+"\n")
                time.sleep(10)
                i+=1
            finally:
                lock.release()
