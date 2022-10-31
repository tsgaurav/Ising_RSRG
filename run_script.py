#!/usr/bin/env python
import RSRG
import time
from mpi4py import MPI

L = 30    #System size of LxL
steps = int(0.9 * L*L)   #Number of decimation steps

a_vals = np.array([0.1])     #Parameters on linear pdf a+bx
b_vals = np.array([0.1])


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

print(result)mport numpy as np

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




J_ij_vals, h_vals, np.array(R0_array) = run_decimation(L, steps, a_vals, b_vals)
