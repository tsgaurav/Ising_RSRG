#!/usr/bin/env python3

from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
import os, csv, sys, pickle, time
import pandas as pd

log_file = pd.read_csv("openBC/bdry_log_file.csv")
start_ind = 0
end_ind = 1
selected_files = log_file[start_ind:end_ind]

def generate_corr_matrix(reverse_dict, L):
    vect = np.zeros(shape=(L*L, L*L), dtype=bool)
    for key in reverse_dict:
        if reverse_dict[key] is None: continue
        vect[key,reverse_dict[key]] = True
    return np.einsum('ab,ac->bc', vect, vect)

def generate_corr_matrix_alt(clust_dict, L):
    corr_mat_alt = np.zeros(shape=(L*L, L*L), dtype=bool)
    for i in range(L*L):
        for j in range(i, L*L):
            corr_mat_alt[i,j] = (clust_dict[i]==clust_dict[j])
    corr_mat_alt = corr_mat_alt.T + corr_mat_alt 
    return corr_mat_alt

def generate_corr_matrix_alt2(clust_dict, reverse_dict, L):
    #Fastest
    corr_mat = np.zeros(shape=(L*L, L*L), dtype=bool)
    for i in range(L*L):
        corr_mat[i, reverse_dict[clust_dict[i]]] = True
    return corr_mat

ts_list = selected_files['ts'].tolist()
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
        n_runs = n_runs_list[nbr]
        L = L_list[nbr]

        with open("openBC/IsingB_2D_clusters_"+str(ts)+".pkl", "rb") as fp:   
            clust_list_final = pickle.load(fp)

        clust_dict_list, reverse_dict_list = [], []

        for core_pair in clust_list_final:
            clust_list_temp = core_pair[0]
            reverse_list_temp = core_pair[1]
            clust_dict_list+=clust_list_temp
            reverse_dict_list+=reverse_list_temp

        mean_corr_mat = np.zeros(shape=(L*L,L*L))
        for instance in range(len(reverse_dict_list)):
            mean_corr_mat += generate_corr_matrix_alt2(clust_dict_list[instance], reverse_dict_list[instance], L)
        mean_corr_mat = mean_corr_mat/n_runs
        
        with open("openBC/IsingB_2D_cmat_"+str(ts)+".pkl", "wb") as fp:   #Pickling
            pickle.dump(mean_corr_mat, fp)
        comm.send("Complete", dest=0, tag=11)
        
        return None
    
def manager(npr, njobs):
    
    multiplier = int(njobs/(npr-1))
    remainder = njobs%(npr-1)

    completion_list = []
    jobcnt = 0
    while jobcnt<njobs:
        send_pr = npr
        if multiplier == 0: send_pr = remainder
        multiplier = multiplier -1

        for i in range(1, send_pr+1):
            nbr = jobcnt
            jobcnt =jobcnt+1
            #print("Sending to process", i)
            comm.send(nbr, dest=i, tag=11)
        
        for i in range(1, send_pr+1):
            data = comm.recv(source=i, tag=11)
            completion_list.append(data)
        
    for i in range(1, npr):
        #print("Terminating process ",i)
        comm.send(-1, dest=i, tag=11)
    
 
main()
