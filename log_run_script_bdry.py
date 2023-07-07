#!/usr/bin/env python3
import sys
#sys.path.insert(0, '')
from mpi4py import MPI
import numpy as np
from aux_funcs import *
from bdry_log_aux_funcs import *
from RSRG_bdry_log_class import *
from copy import deepcopy
import pandas as pd
import time, pickle, csv, os, json, fasteners


comm = MPI.COMM_WORLD
rank = comm.Get_rank() # get your process ID
n_processes = comm.size

L = int(sys.argv[1])
steps = L*L - 20#int(0.992*L*L)
a_mat = np.array([[0.1, 0.1],[0.1, 0.1]])
b_mat = np.array([[0.105, 0.105],[0.105, 0.105]])

w_blk = float(sys.argv[2])
w_bdry = float(sys.argv[3])
w_mixed = float(sys.argv[4])

w_mat = np.array([[w_blk, w_mixed],[w_mixed, w_bdry]])

#ind_dict, adj_ind, bdry_dict = triangle_lattice_boundary_dictionary(L)
#bdry_dict[-L:] = False
ind_dict, adj_ind = triangle_lattice_dictionary(L)
nn_ind = triangle_nn_indices(L)
adj_ind = triangle_nn_indices(L)
bdry_dict = np.zeros(L*L, dtype=bool)

measure_list = L*L - gen_check_list(L*L, steps, 20, 0.1)

out_dir = "log_bdry_output/PBC_test/"

#cluster_dict_list = [np.array([]) for step in range(len(measure_list))]

n_runs = 5

input_dict = {"L":L, "steps":steps,"measure_list":measure_list,'w_blk':w_blk, 'w_bdry':w_bdry,'w_mixed':w_mixed, "n_runs":n_runs*n_processes}
#input_dict['misc_notes'] = 'Only top is bdry'
#sampler = RandomLinDistWidthSampler(a_mat, b_mat, w_mat, L*2*n_runs)

if rank == 0: # The master is the only process that reads the file
	t0 = time.time()
	ts = str(int(100*time.time()))[4:]
	ts+= str(L)+str(10*w_bdry)+str(10*w_blk)
	data = [[ts]]*n_processes
	with open(out_dir+"LIsingB_2D_output_"+ts+".txt", "w") as writer:
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
bdry_moment_list = []
blk_moment_list = []
active_clust_list = []

for item in data:  #Sending to processes
	for inst in range(n_runs):  #Within each process
        
		#zeta_ij_vals = fill_zeta_ij_bdry(L*L, bdry_dict, adj_ind, a_mat, b_mat, w_mat)
		#zeta_ij_vals = fill_zeta_ij_bdry_v2(L*L, bdry_dict,adj_ind, sampler)
		zeta_ij_vals = fill_zeta_ij_matrix_width(L*L, adj_ind, 0.1, 0.105, w_blk)
		beta_vals = fill_beta_vals_bdry(L*L, bdry_dict, 1.,1.)
        
		test = bdry_log_system(L*L, deepcopy(adj_ind), deepcopy(bdry_dict), zeta_ij_vals, beta_vals, track_moments=False)

		check_acc = 0
		for i in range(steps+1):
			test.decimate()
			if i in measure_list: 
				beta_remain_blk = test.beta_vals[~test.bdry_dict]
				beta_remain_blk = beta_remain_blk[beta_remain_blk!=0]

				beta_remain_bdry = test.beta_vals[test.bdry_dict]
				beta_remain_bdry = beta_remain_bdry[beta_remain_bdry!=0]

				mask = test.zeta_ij_vals>=10	
				zeta_remain = deepcopy(test.zeta_ij_vals)
				zeta_remain[mask] = 0
				zeta_remain.eliminate_zeros()
								
				zeta_remain_blk_blk = sparse.triu(zeta_remain[~test.bdry_dict,:][:,~test.bdry_dict]).data
				zeta_remain_bdry_bdry = sparse.triu(zeta_remain[test.bdry_dict,:][:,test.bdry_dict]).data              
				zeta_remain_blk_bdry = sparse.triu(zeta_remain[~test.bdry_dict,:][:,test.bdry_dict]).data

				lock = fasteners.InterProcessLock('/tmp/tmplockfile'+item)
				gotten = lock.acquire(blocking=True)
				if gotten:
					try:
						with open(out_dir+"LIsingB_2D_output_"+item+".txt", "a") as writer:
							i_num = f"{rank:02}" + f"{inst:02}"
							writer.write("In"+i_num+"_hbl_m"+f"{check_acc:02}")   #hbl = h bulk
							json.dump(beta_remain_blk.tolist(), writer)
							writer.write('\n')
							writer.write("In"+i_num+"_hbd_m"+f"{check_acc:02}")    #hbd = h bdry
							json.dump(beta_remain_bdry.tolist(), writer)
							writer.write('\n')
							#writer.write("In"+i_num+"_crd_m"+f"{check_acc:02}")    #crd = cluster reverse dictionary
							#json.dump(test.reverse_dict, writer)
							#writer.write('\n')
							#writer.write("In"+i_num+"_cbd_m"+f"{check_acc:02}")     #cbd = cluster boundary dictionary
							#json.dump(test.bdry_dict.tolist(), writer)
							#writer.write('\n')
							writer.write("In"+i_num+"_Jll_m"+f"{check_acc:02}")   #Jll = J bulk-bulk
							json.dump(zeta_remain_blk_blk.tolist(), writer)
							writer.write('\n')
							writer.write("In"+i_num+"_Jrr_m"+f"{check_acc:02}")    #Jrr = J bdry-bdry
							json.dump(zeta_remain_bdry_bdry.tolist(), writer)
							writer.write('\n')
							writer.write("In"+i_num+"_Jlr_m"+f"{check_acc:02}")   #Jlr = J bulk-bdry
							json.dump(zeta_remain_blk_bdry.tolist(), writer)
							writer.write('\n')
					finally:
						lock.release()
				check_acc+=1
		cluster_dict_list.append(test.clust_dict)
		reverse_clust_dict_list.append(test.reverse_dict)
		bdry_dict_list.append(test.bdry_dict)
		active_clust_list.append(test.active_clust_list)
		bdry_moment_list.append(test.bdry_moment_list)
		blk_moment_list.append(test.blk_moment_list)

#data = (h_dist_list_blk, h_dist_list_bdry, Omega_list_composite, decimation_type_composite)
clust_data = [cluster_dict_list, reverse_clust_dict_list, bdry_dict_list, active_clust_list, bdry_moment_list, blk_moment_list]

# Send the results back to the master processes


#processed_data = comm.gather(data,root=0)
clust_list_final = comm.gather(clust_data, root=0)


if rank == 0:

    #J_dist_list = np.concatenate(J_dist_list_proc, axis=0)
    
    ts_final = time.time()
    
    input_dict['runtime']= ts_final - t0

    with open(out_dir+"LIsingB_2D_clusters_"+ts+".pkl", "wb") as fp:   #Pickling
        pickle.dump(clust_list_final, fp)

    #with open("output/Ising_2D_J_dist_"+ts+".pkl", "wb") as fp:   #Pickling
        #pickle.dump(J_dist_list, fp)

    with open(out_dir+"LIsingB_2D_input_"+ts+".pkl", "wb") as fp:
        pickle.dump(input_dict, fp)
    
    input_dict['ts'] = ts
    input_dict.pop('measure_list')

    if not os.path.exists(out_dir+"bdry_log_file.csv"):
        with open(out_dir+"bdry_log_file.csv", 'w') as csv_file:
            header = list(input_dict.keys())
            row = list(input_dict.values())

            csv_writer = csv.writer(csv_file, lineterminator='\n')
            csv_writer.writerow(header)
            csv_writer.writerow(row)

    else:
        with open(out_dir+"bdry_log_file.csv", 'a') as csv_file:
            row = list(input_dict.values())
            csv_writer = csv.writer(csv_file, lineterminator='\n')
            csv_writer.writerow(row)



