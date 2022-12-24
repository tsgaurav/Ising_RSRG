#!/usr/bin/env python
from mpi4py import MPI
import numpy as np
from aux_funcs import *
from RSRG import *
from RSRG_class import *
from copy import deepcopy
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank() # get your process ID
n_processes = comm.size


L = 30
steps = int(0.95*L*L)
measure_step = 20
a, b = 0.1, 0.1
ind_dict, adj_ind = triangle_lattice_dictionary(L)
nn_ind = triangle_nn_indices(L)
nnn_ind = triangle_nnn_indices(L)

n_runs = 1


data = [[1]]*n_processes   # init the data    

#if rank == 0: # The master is the only process that reads the file
#    data = data# something read from file

# Divide the data among processes
data = comm.scatter(data, root=0)
index = 0

R0_array_sum = np.zeros(shape=(n_processes, int(np.ceil(steps/measure_step - 1))))
for item in data:  #Sending to processes
    for inst in range(n_runs):  #Within each process
        J_ij_vals = fill_J_ij_matrix(L*L, nn_ind, nnn_ind, a, b)
        h_vals = np.exp(-np.random.exponential(size=L*L))
        test = system(L*L, deepcopy(adj_ind), J_ij_vals, h_vals) 
        for i in range(steps):
            test.decimate()
        
        R0_array_sum[index,:] += np.array(test.R0_array)
        index += 1
# Send the results back to the master processes
newData = comm.gather(R0_array_sum,root=0)

np.save("output/Ising_2D_"+str(int(time.time())), R0_array_sum.sum(axis=0))
