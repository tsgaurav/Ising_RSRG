#!/usr/bin/env python3

from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
import os, csv, sys, pickle, time
import pandas as pd


run_dir = "mag_scaling/"
log_file = pd.read_csv(run_dir+"log_file.csv")
start_ind = 0
end_ind = 4
zone_width = 0
selected_files = log_file[start_ind:end_ind]

def check_percolation(clust_dict, L, width):
    end_point = int(L/2)
    for y1 in range(L):
        for y1_width in range(-width, width+1):
            for y2 in range(L):
                for y2_width in range(-width, width+1):
                    start_ind = index_map(y1_width%L, y1, L)
                    end_index = index_map((end_point+y2_width)%L, y2, L)
                    if (clust_dict[start_ind] == clust_dict[end_index]): 
                        return True
    return False

def index_map(x, y, L):
    if x>L or y>L: return None
    return x*L + y


ts_list = selected_files['ts'].tolist()
w_list = selected_files['w'].tolist()
n_runs_list = selected_files['n_runs'].tolist()
L_list = selected_files['L'].tolist()

comm = MPI.COMM_WORLD

rank = comm.Get_rank()            #number of the process running the code
size = comm.Get_size()  #total number of processes running

N = len(ts_list)

def main():
    if (rank==0):
        manager(size, N)
    else:
        worker(rank)
        
def worker(i):
    while True:
        nbr = comm.recv(source=0, tag=11)
        if nbr==-1: break
        
        ts = ts_list[nbr]
        w_val = w_list[nbr]
        n_runs = n_runs_list[nbr]
        L = L_list[nbr]
        #print(nbr, ts, L)
        with open(run_dir+"LIsing_2D_clusters_"+str(ts)+".pkl", "rb") as fp:   
            clust_list_final = pickle.load(fp)

        clust_dict_list, reverse_dict_list = [], []

        for core_pair in clust_list_final:
            clust_list_temp = core_pair[0]
            reverse_list_temp = core_pair[1]
            clust_dict_list+=clust_list_temp
            reverse_dict_list+=reverse_list_temp

        perc_prob_temp = np.array([check_percolation(clust, L, zone_width) for clust in clust_dict_list])
        print(L, w_val, perc_prob_temp.mean(), perc_prob_temp.std()/np.sqrt(n_runs-1))
        comm.send([L, w_val, perc_prob_temp.mean(), perc_prob_temp.std()/np.sqrt(n_runs-1)], dest=0, tag=11)
        
        return None
    
def manager(npr, njobs):
    
    multiplier = int(njobs/(npr-1))
    remainder = njobs%(npr-1)

    completion_list = []
    jobcnt = 0
    while jobcnt<njobs:
        send_pr = npr
        if multiplier == 0: send_pr = remainder + 1
        multiplier = multiplier -1

        for i in range(1, send_pr):
            nbr = jobcnt
            jobcnt =jobcnt+1
            #print("Sending to process", i)
            comm.send(nbr, dest=i, tag=11)
        
        for i in range(1, send_pr):
            data = comm.recv(source=i, tag=11)
            completion_list.append(data)
    
    for i in range(1, npr):
        #print("Terminating process ",i)
        comm.send(-1, dest=i, tag=11)
    
 
main()
