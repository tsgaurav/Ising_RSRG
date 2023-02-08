#!/usr/bin/env python
from mpi4py import MPI
import numpy as np
from aux_funcs import *
from RSRG import *
from RSRG_class import *
from copy import deepcopy
import time
import pickle

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

measure_list = gen_check_list(L*L, steps-1, 20)

J_dist_list = [np.array([]) for step in range(len(measure_list))]
h_dist_list = [np.array([]) for step in range(len(measure_list))]

n_runs = 10

input_dict = {"L":L, "steps":steps,"measure_list":measure_list,"(a,b)":(a,b), "n_runs":n_runs*n_processes}

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
        check_acc = 0
        for i in range(steps):
            test.decimate()
            if i in measure_list: 
                h_remain = test.h_vals[test.h_vals!=0]
                h_dist_list[check_acc] = np.concatenate((h_dist_list[check_acc],-np.log(h_remain/test.Omega)))

                J_remain = -np.log(sparse.find(test.J_ij_vals)[2]) + np.log(test.Omega)
                J_dist_list[check_acc] = np.concatenate((J_dist_list[check_acc], J_remain))

                check_acc+=1
        
# Send the results back to the master processes
newData = comm.gather(R0_array_sum,root=0)

#np.save("output/Ising_2D_"+str(int(time.time())), R0_array_sum.sum(axis=0))

ts = str(int(time.time()))

with open("output/Ising_2D_h_dist_"+ts, "wb") as fp:   #Pickling
    pickle.dump(h_dist_list, fp)

with open("output/Ising_2D_J_dist_"+ts, "wb") as fp:   #Pickling
    pickle.dump(J_dist_list, fp)

with open("output/Ising_2D_input_"+ts, "wb") as fp:
    pickle.dump(input_dict, fp)
