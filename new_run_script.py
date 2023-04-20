#!/usr/bin/env python3
from mpi4py import MPI
import numpy as np
from aux_funcs import *
from RSRG import *
from RSRG_class import *
from copy import deepcopy
import pandas as pd
import time, pickle, sys, csv, os


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

comm = MPI.COMM_WORLD

rank = comm.Get_rank()            #number of the process running the code
size = comm.Get_size()  #total number of processes running
N = 5
n_runs = (size-1)*(N+1)

input_dict = {"L":L, "steps":steps,"measure_list":measure_list,'a':a, 'b':b,'w':w, "n_runs":n_runs}

def main():

    if (rank == 0) :
        manager(size, size*N)
    else:
        worker(rank)

def worker(i):
    while True:
        nbr = comm.recv(source=0, tag =11)
        if nbr == -1: break
        J_ij_vals = fill_J_ij_matrix_width(L*L, nn_ind, a, b, w)
        h_vals = np.exp(-np.random.exponential(size=L*L))
        test = system(L*L, deepcopy(nn_ind), J_ij_vals, h_vals)
        check_acc = 0
        for i in range(steps):
            test.decimate()
            if i in measure_list:
                continue
                #h_remain = test.h_vals[test.h_vals!=0]
                #h_dist_list[check_acc] = np.concatenate((h_dist_list[check_acc],-np.log(h_remain/test.Omega)))

                #J_remain = -np.log(sparse.find(test.J_ij_vals)[2]) + np.log(test.Omega)
                #J_dist_list[check_acc] = np.concatenate((J_dist_list[check_acc], J_remain))

                #check_acc+=1
        #Omega_list_composite = np.concatenate((Omega_list_composite, np.array(test.Omega_array)))
        #decimation_type_composite = np.concatenate((decimation_type_composite, np.array(test.coupling_dec_list, dtype=bool)))

        result = [test.clust_dict, test.reverse_dict]
        comm.send(result, dest=0, tag=11)

def manager(npr, njobs):
    clust_dict_list = []
    reverse_dict_list =  []
    jobcnt = 0
    while jobcnt < njobs:
        
        for i in range(1, npr):
            jobcnt = jobcnt +1
            nbr = 1 + (jobcnt % (npr-1))
            #print('Manager sending', jobcnt,'worker', i)
            comm.send(nbr, dest=i, tag=11)

        for i in range(1, npr):
            data = comm.recv(source=i, tag = 11)
            #print('Manager received',data,'worker',i)
            clust_dict_list += data[0]
            reverse_dict_list +=data [1]
    for i in range(1, npr):
        #print('Kill worker',i)
        comm.send(-1, dest=i, tag=11)
    clust_list_final = [clust_dict_list, reverse_dict_list]
    ts = str(int(time.time()))

    #with open("output/Ising_2D_output_"+ts+".pkl", "wb") as fp:   #Pickling
    #    pickle.dump(processed_data, fp)

    with open("output/Ising_2D_clusters_"+ts+".pkl", "wb") as fp:   #Pickling
        pickle.dump(clust_list_final, fp)

    with open("output/Ising_2D_input_"+ts+".pkl", "wb") as fp:
        pickle.dump(input_dict, fp)

    input_dict['ts'] = ts
    input_dict.pop('measure_list')

    if not os.path.exists("output/log_file.csv"):
        with open("output/log_file.csv", 'w') as csv_file:
            header = list(input_dict.keys())
            row = list(input_dict.values())

            csv_writer = csv.writer(csv_file, lineterminator='\n')
            csv_writer.writerow(header)
            csv_writer.writerow(row)

    else:
        with open("output/log_file.csv", 'a') as csv_file:
            row = list(input_dict.values())
            csv_writer = csv.writer(csv_file, lineterminator='\n')
            csv_writer.writerow(row)

main()
