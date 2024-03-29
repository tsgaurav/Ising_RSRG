#!/usr/bin/env python3
from mpi4py import MPI
import numpy as np
from aux_funcs import *
from RSRG import *
from RSRG_class import *
from copy import deepcopy
import pandas as pd
import time, pickle, sys, csv, os, json

comm = MPI.COMM_WORLD
rank = comm.Get_rank() # get your process ID
n_processes = comm.size


out_dir = "output/fix_test_2/" 
L = int(sys.argv[2])
steps = L*L - 20 #int(0.992*L*L)
measure_step = 20
a = float(sys.argv[3])
b = 0.105
w = float(sys.argv[1])
track_moments = True
ind_dict, adj_ind = triangle_lattice_dictionary(L)
nn_ind = triangle_nn_indices(L)
nnn_ind = triangle_nnn_indices(L)

measure_list = L*L - gen_check_list(L*L, steps, 20)
 

#cluster_dict_list = [np.array([]) for step in range(len(measure_list))]

n_runs = 10

input_dict = {"L":L, "steps":steps,"measure_list":measure_list,'a':a, 'b':b,'w':w, "n_runs":n_runs*n_processes}


if rank == 0: # The master is the only process that reads the file
	ts = str(int(100*time.time()+100*np.random.random()))[2:]
	data = [[ts]]*n_processes
	with open(out_dir+"Ising_2D_output_"+ts+".txt", "w") as writer:
		writer.write("##Output"+'\n')

else:
    data = None


#Divide the data among processes
data = comm.scatter(data, root=0)
index = 0


#J_dist_list = [np.array([]) for step in range(len(measure_list))]
#h_dist_list = [np.array([]) for step in range(len(measure_list))]


#Omega_list_composite = np.array([])
#decimation_type_composite = np.array([], dtype=bool)

cluster_dict_list = []
reverse_clust_dict_list = []
moment_list_list = []

for item in data:  #Sending to processes
	for inst in range(n_runs):  #Within each process
		J_ij_vals = fill_J_ij_matrix_width(L*L, nn_ind, a, b, w)
		h_vals = np.exp(-np.random.exponential(size=L*L))
		test = system(L*L, deepcopy(nn_ind), J_ij_vals, h_vals, track_moments=track_moments)
		for i in range(steps+1):
			test.decimate()
			if i in measure_list:
				h_remain = test.h_vals[test.h_vals!=0]
				J_remain = -np.log(sparse.triu(test.J_ij_vals).data) + np.log(test.Omega)

				with open(out_dir+"Ising_2D_output_"+item+".txt", "a") as writer:
					i_num = f"{rank:02}" + f"{inst:02}"
					writer.write("In"+i_num+"_h_m"+f"{i:02}")
					json.dump((-np.log(h_remain/test.Omega)).tolist(), writer)
					writer.write('\n')
					writer.write("In"+i_num+"_J_m"+f"{i:02}")
					json.dump(J_remain.tolist(), writer)
					writer.write('\n')

		cluster_dict_list.append(test.clust_dict)
		reverse_clust_dict_list.append(test.reverse_dict)
		moment_list_list.append(test.moment_list)
#data = (J_dist_list, h_dist_list, Omega_list_composite, decimation_type_composite)
clust_data = [cluster_dict_list, reverse_clust_dict_list, moment_list_list]

# Send the results back to the master processes


#processed_data = comm.gather(data,root=0)
clust_list_final = comm.gather(clust_data, root=0)


if rank == 0:

    #J_dist_list = np.concatenate(J_dist_list_proc, axis=0)
    
    #with open("output/Ising_2D_output_"+ts+".pkl", "wb") as fp:   #Pickling
    #    pickle.dump(processed_data, fp)

    with open(out_dir+"Ising_2D_clusters_"+ts+".pkl", "wb") as fp:   #Pickling
        pickle.dump(clust_list_final, fp)

    with open(out_dir+"Ising_2D_input_"+ts+".pkl", "wb") as fp:
        pickle.dump(input_dict, fp)
    
    input_dict['ts'] = ts
    input_dict.pop('measure_list')

    if not os.path.exists(out_dir+"log_file.csv"):
        with open(out_dir+"log_file.csv", 'w') as csv_file:
            header = list(input_dict.keys())
            row = list(input_dict.values())

            csv_writer = csv.writer(csv_file, lineterminator='\n')
            csv_writer.writerow(header)
            csv_writer.writerow(row)

    else:
        with open(out_dir+"log_file.csv", 'a') as csv_file:
            row = list(input_dict.values())
            csv_writer = csv.writer(csv_file, lineterminator='\n')
            csv_writer.writerow(row)



