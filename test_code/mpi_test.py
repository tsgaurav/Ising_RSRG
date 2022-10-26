#!/usr/bin/env python
from mpi4py import MPI
import numpy as np

def myFun(x):
    return x+2
    # simple example, the real one would be complicated

comm = MPI.COMM_WORLD
rank = comm.Get_rank() # get your process ID
data = [np.arange(3)]*2   # init the data    

if rank == 0: # The master is the only process that reads the file
    data = [np.arange(2)]*2# something read from file

# Divide the data among processes
data = comm.scatter(data, root=0)

result = []
for item in data:
    result.append(myFun(item))

# Send the results back to the master processes
newData = comm.gather(result,root=0)

print(result)
