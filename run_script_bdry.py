#!/usr/bin/env python3
import sys
#sys.path.insert(0, '')
from mpi4py import MPI
import numpy as np
from aux_funcs import *
from bdry_aux_funcs import *
from RSRG_bdry_class import *
from copy import deepcopy
import pandas as pd
import time, pickle, csv, os, json


comm = MPI.COMM_WORLD
rank = comm.Get_rank() # get your process ID
n_processes = comm.size

L = int(sys.argv[1])
steps = int(0.992*L*L)
a_mat = np.array([[0.1, 0.1],[0.1, 0.1]])
b_mat = np.array([[0.105, 0.105],[0.105, 0.105]])

w_blk = float(sys.argv[2])
w_bdry = float(sys.argv[3])
w_mixed = float(sys.argv[4])

w_mat = np.array([[w_blk, w_mixed],[w_mixed, w_blk]])

ind_dict, adj_ind, bdry_dict = triangle_lattice_boundary_dictionary(L)


measure_list = L*L - gen_check_list(L*L, steps, 20)
 

#cluster_dict_list = [np.array([]) for step in range(len(measure_list))]

n_runs = 30

input_dict = {"L":L, "steps":steps,"measure_list":measure_list,'w_blk':w_blk, 'w_bdry':w_bdry,'w_mixed':w_mixed, "n_runs":n_runs*n_processes}


if rank == 0: # The master is the only process that reads the file
	ts = str(int(100*time.time()+100*np.random.random()))[2:]
	data = [[ts]]*n_processes
	with open("bdry_output/IsingB_2D_output_"+ts+".txt", "w") as writer:
		writer.write("##Output"+'\n')

else:
    data = None


#Divide the data among processes
data = comm.scatter(data, root=0)
index = 0


#J_dist_list_blk = [np.array([]) for step in range(len(measure_list))]
#h_dist_list_blk = [np.array([]) for step in range(len(measure_list))]

#J_dist_list_bdry = [np.array([]) for step in range(len(measure_list))]
#h_dist_list_bdry = [np.array([]) for step in range(len(measure_list))]

#Omega_list_composite = np.array([])
#decimation_type_composite = np.array([], dtype=bool)

cluster_dict_list = []
reverse_clust_dict_list = []
bdry_dict_list = []

for item in data:  #Sending to processes
	for inst in range(n_runs):  #Within each process
        
		J_ij_vals = fill_J_ij_bdry(L*L, bdry_dict, adj_ind, a_mat, b_mat, w_mat)
		h_vals = fill_h_vals_bdry(L*L, bdry_dict, 1.,1.)
        
		test = boundary_system(L*L, deepcopy(adj_ind), deepcopy(bdry_dict), J_ij_vals, h_vals)

		check_acc = 0
		for i in range(steps+1):
			test.decimate()
			if i in measure_list: 
				h_remain_blk = test.h_vals[~bdry_dict]
				h_remain_blk = h_remain_blk[h_remain_blk!=0]

				h_remain_bdry = test.h_vals[bdry_dict]
				h_remain_bdry = h_remain_bdry[h_remain_bdry!=0]
                
				with open("bdry_output/IsingB_2D_output_"+item+".txt", "a") as writer:
					i_num = f"{rank:02}" + f"{inst:02}"
					writer.write("In"+i_num+"_hbl_m"+f"{check_acc:02}")   #hbl = h bulk
					json.dump((-np.log(h_remain_blk/test.Omega)).tolist(), writer)
					writer.write('\n')
					writer.write("In"+i_num+"_hbd_m"+f"{check_acc:02}")    #hbd = h bdry
					json.dump((-np.log(h_remain_bdry/test.Omega)).tolist(), writer)
					writer.write('\n')

                #Need to figure out how to read off the bond matrix, since we have blk-blk, blk-bdry and bdry-bdry couplings

				check_acc+=1
		cluster_dict_list.append(test.clust_dict)
		reverse_clust_dict_list.append(test.reverse_dict)
		bdry_dict_list.append(test.bdry_dict)

#data = (h_dist_list_blk, h_dist_list_bdry, Omega_list_composite, decimation_type_composite)
clust_data = [cluster_dict_list, reverse_clust_dict_list, bdry_dict_list]

# Send the results back to the master processes


#processed_data = comm.gather(data,root=0)
clust_list_final = comm.gather(clust_data, root=0)


if rank == 0:

    #J_dist_list = np.concatenate(J_dist_list_proc, axis=0)
    
    #ts = str(int(time.time()))
    
    #with open("bdry_output/IsingB_2D_output_"+ts+".pkl", "wb") as fp:   #Pickling
    #    pickle.dump(processed_data, fp)

    with open("bdry_output/IsingB_2D_clusters_"+ts+".pkl", "wb") as fp:   #Pickling
        pickle.dump(clust_list_final, fp)

    #with open("output/Ising_2D_J_dist_"+ts+".pkl", "wb") as fp:   #Pickling
        #pickle.dump(J_dist_list, fp)

    with open("bdry_output/IsingB_2D_input_"+ts+".pkl", "wb") as fp:
        pickle.dump(input_dict, fp)
    
    input_dict['ts'] = ts
    input_dict.pop('measure_list')

    if not os.path.exists("bdry_output/bdry_log_file.csv"):
        with open("bdry_output/bdry_log_file.csv", 'w') as csv_file:
            header = list(input_dict.keys())
            row = list(input_dict.values())

            csv_writer = csv.writer(csv_file, lineterminator='\n')
            csv_writer.writerow(header)
            csv_writer.writerow(row)

    else:
        with open("bdry_output/bdry_log_file.csv", 'a') as csv_file:
            row = list(input_dict.values())
            csv_writer = csv.writer(csv_file, lineterminator='\n')
            csv_writer.writerow(row)



