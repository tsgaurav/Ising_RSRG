#!/usr/bin/env python3
from mpi4py import MPI
import numpy as np
from aux_funcs import *
from RSRG import *
from RSRG_class import *
from copy import deepcopy
import time
import pickle
import sys

comm = MPI.COMM_WORLD
rank = comm.Get_rank() # get your process ID
n_processes = comm.size

L = 70
steps = int(0.95*L*L)
measure_step = 20
a, b = 0.1, 0.105
w = float(sys.argv[1])
ind_dict, adj_ind = triangle_lattice_dictionary(L)
nn_ind = triangle_nn_indices(L)
nnn_ind = triangle_nnn_indices(L)

measure_list = gen_check_list(L*L, steps-1, 20)


#cluster_dict_list = [np.array([]) for step in range(len(measure_list))]

n_runs = 16

input_dict = {"L":L, "steps":steps,"measure_list":measure_list,"(a,b,w)":(a,b,w), "n_runs":n_runs*n_processes}


if rank == 0: # The master is the only process that reads the file
    #data = data# something read from file
    data = [[1]]*n_processes

else:
    data = None


#Divide the data among processes
data = comm.scatter(data, root=0)
index = 0


J_dist_list = [np.array([]) for step in range(len(measure_list))]
h_dist_list = [np.array([]) for step in range(len(measure_list))]


Omega_list_composite = np.array([])
decimation_type_composite = np.array([], dtype=bool)

cluster_dict_list = []
reverse_clust_dict_list = []

for item in data:  #Sending to processes
    for inst in range(n_runs):  #Within each process
        
        #J_ij_vals = fill_J_ij_matrix(L*L, nn_ind, nnn_ind, a, b)
        J_ij_vals = fill_J_ij_matrix_width(L*L, nn_ind, a, b, w)
        h_vals = np.exp(-np.random.exponential(size=L*L))
        test = system(L*L, deepcopy(nn_ind), J_ij_vals, h_vals)
        check_acc = 0
        for i in range(steps):
            test.decimate()
            if i in measure_list: 
                h_remain = test.h_vals[test.h_vals!=0]
                h_dist_list[check_acc] = np.concatenate((h_dist_list[check_acc],-np.log(h_remain/test.Omega)))

                J_remain = -np.log(sparse.find(test.J_ij_vals)[2]) + np.log(test.Omega)
                J_dist_list[check_acc] = np.concatenate((J_dist_list[check_acc], J_remain))

                check_acc+=1
        Omega_list_composite = np.concatenate((Omega_list_composite, np.array(test.Omega_array)))
        decimation_type_composite = np.concatenate((decimation_type_composite, np.array(test.coupling_dec_list, dtype=bool)))
        cluster_dict_list.append(test.clust_dict)
        reverse_clust_dict_list.append(test.reverse_dict)
data = (J_dist_list, h_dist_list, Omega_list_composite, decimation_type_composite)
clust_data = [cluster_dict_list, reverse_clust_dict_list]

# Send the results back to the master processes


processed_data = comm.gather(data,root=0)
clust_list_final = comm.gather(clust_data, root=0)

#J_dist_list_proc = comm.gather(J_dist_list, root=0)
#h_dist_list_proc = comm.gather(h_dist_list, root=0)
#Omega_list_composite_proc = comm.gather(Omega_list_composite, root=0)
#decimation_type_composite_proc = comm.gather(decimation_type_composite, root=0)

#with open("test.pkl", "wb") as fp:
#    pickle.dump(comm.gather(data, root=0), fp)
#np.save("output/Ising_2D_"+str(int(time.time())), R0_array_sum.sum(axis=0))
if rank == 0:

    #J_dist_list = np.concatenate(J_dist_list_proc, axis=0)
    
    ts = str(int(time.time()))
    
    with open("output/Ising_2D_output_"+ts+".pkl", "wb") as fp:   #Pickling
        pickle.dump(processed_data, fp)

    with open("output/Ising_2D_clusters_"+ts+".pkl", "wb") as fp:   #Pickling
        pickle.dump(clust_list_final, fp)

    #with open("output/Ising_2D_J_dist_"+ts+".pkl", "wb") as fp:   #Pickling
        #pickle.dump(J_dist_list, fp)

    with open("output/Ising_2D_input_"+ts+".pkl", "wb") as fp:
        pickle.dump(input_dict, fp)
