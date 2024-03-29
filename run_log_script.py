#!/usr/bin/env python3
from mpi4py import MPI
import numpy as np
from aux_funcs import *
from RSRG import *
from RSRG_class import *
from copy import deepcopy
import pandas as pd
import time, pickle, sys, csv, os

comm = MPI.COMM_WORLD
rank = comm.Get_rank() # get your process ID
n_processes = comm.size

L = int(sys.argv[2])
steps = int(0.992*L*L)
measure_step = 20
a = float(sys.argv[3])
b = 0.105
w = float(sys.argv[1])
ind_dict, adj_ind = triangle_lattice_dictionary(L)
nn_ind = triangle_nn_indices(L)
nnn_ind = triangle_nnn_indices(L)

measure_list = gen_check_list(L*L, steps-1, 20)
 

#cluster_dict_list = [np.array([]) for step in range(len(measure_list))]

n_runs = 10

input_dict = {"L":L, "steps":steps,"measure_list":measure_list,'a':a, 'b':b,'w':w, "n_runs":n_runs*n_processes}


if rank == 0: # The master is the only process that reads the file
    #data = data# something read from file
    data = [[1]]*n_processes

else:
    data = None


#Divide the data among processes
data = comm.scatter(data, root=0)
index = 0


zeta_dist_list = [np.array([]) for step in range(len(measure_list))]
beta_dist_list = [np.array([]) for step in range(len(measure_list))]


Gamma_list_composite = np.array([])
decimation_type_composite = np.array([], dtype=bool)

cluster_dict_list = []
reverse_clust_dict_list = []

for item in data:  #Sending to processes
    for inst in range(n_runs):  #Within each process
        
        #J_ij_vals = fill_J_ij_matrix(L*L, nn_ind, nnn_ind, a, b)
        zeta_ij_vals = fill_zeta_ij_matrix_width(L*L, nn_ind, a, b, w)
        beta_vals = np.random.exponential(size=L*L)
        test = log_system(L*L, deepcopy(nn_ind), zeta_ij_vals, beta_vals)
        check_acc = 0
        for i in range(steps):
            test.decimate()
            if i in measure_list: 
                beta_remain = test.beta_vals[test.beta_vals!=0]
                beta_dist_list[check_acc] = np.concatenate((beta_dist_list[check_acc],beta_remain-test.Gamma))

                zeta_remain = sparse.find(test.zeta_ij_vals)[2] - test.Gamma
                zeta_dist_list[check_acc] = np.concatenate((zeta_dist_list[check_acc], zeta_remain))

                check_acc+=1
        Gamma_list_composite = np.concatenate((Gamma_list_composite, np.array(test.Gamma_array)))
        decimation_type_composite = np.concatenate((decimation_type_composite, np.array(test.coupling_dec_list, dtype=bool)))
        cluster_dict_list.append(test.clust_dict)
        reverse_clust_dict_list.append(test.reverse_dict)
data = (zeta_dist_list, beta_dist_list, Gamma_list_composite, decimation_type_composite)
clust_data = [cluster_dict_list, reverse_clust_dict_list]

# Send the results back to the master processes


processed_data = comm.gather(data,root=0)
clust_list_final = comm.gather(clust_data, root=0)


if rank == 0:

    #J_dist_list = np.concatenate(J_dist_list_proc, axis=0)
    
    ts = str(int(time.time()))
    
    with open("output/LogIsing_2D_output_"+ts+".pkl", "wb") as fp:   #Pickling
        pickle.dump(processed_data, fp)

    with open("output/LogIsing_2D_clusters_"+ts+".pkl", "wb") as fp:   #Pickling
        pickle.dump(clust_list_final, fp)

    #with open("output/Ising_2D_J_dist_"+ts+".pkl", "wb") as fp:   #Pickling
        #pickle.dump(J_dist_list, fp)

    with open("output/LogIsing_2D_input_"+ts+".pkl", "wb") as fp:
        pickle.dump(input_dict, fp)
    
    input_dict['ts'] = ts
    input_dict.pop('measure_list')

    if not os.path.exists("output/Loglog_file.csv"):
        with open("output/Loglog_file.csv", 'w') as csv_file:
            header = list(input_dict.keys())
            row = list(input_dict.values())

            csv_writer = csv.writer(csv_file, lineterminator='\n')
            csv_writer.writerow(header)
            csv_writer.writerow(row)

    else:
        with open("output/Loglog_file.csv", 'a') as csv_file:
            row = list(input_dict.values())
            csv_writer = csv.writer(csv_file, lineterminator='\n')
            csv_writer.writerow(row)



