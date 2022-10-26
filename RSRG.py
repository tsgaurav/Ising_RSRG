#!/usr/bin/env python
from aux_funcs import *
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


L = 100
steps = 6000


a_vals = np.array([0.1])#np.arange(0.05, 0.2, 0.02)
b_vals = np.array([0.105])#np.arange(0.1,0.3,0.03)




def run_decimation(L, steps, measure_step, a_vals, b_vals, track_moments=False):
    Gamma_array = np.zeros(shape=(len(a_vals), len(b_vals), steps))
    R0_array = []
    mu_array = []
    #R0_array_err = []

    for l, a in enumerate(a_vals):
        for m, b in enumerate(b_vals):

            ind_dict, adj_ind = triangle_lattice_dictionary(L)
            
            
            J_ij_vals = fill_J_ij_matrix(L*L, adj_ind, a, b)
            h_vals = np.exp(-np.random.exponential(size=L*L))
            if track_moments: cluster_tracker = np.ones(L*L)
            
            Omega_0 = max(h_vals.max(), J_ij_vals.max())
            for step in range(steps):

                #if step%1000 == 0: print ("Step: "+str(step)+"/"+str(steps))
                Omega = max(h_vals.max(), J_ij_vals.max())
                Gamma_array[l,m,step] = (np.log(Omega_0/Omega))
                #J_ij_vals = resparse(J_ij_vals, L*L, Omega*(1-steps/(L*L)))

                if Omega == J_ij_vals.max():
                    """
                    Strongest term is an Ising coupling -> Renormalizes to a field given by h_i * h_j/J_ij
                    Delete one of the elements (say h_i), shortening h_vals
                    Update adjacency of j to include both adj(i) and adj(j)
                    Set new Jnew_jk = max (J_jk, J_ik)
                    Delete the corresponding row and column in J_ij (J_i* and J*i) and set the adjacency of i to []
                    """
                    r_ind, c_ind, J_ij = sparse.find(J_ij_vals) 
                    J_ind = np.where(J_ij == Omega)[0][0]
                    i, j = r_ind[J_ind], c_ind[J_ind]

                    h_vals[i] = h_vals[i]*h_vals[j]/Omega    #Might need to include a factor of 2 here since we are symmetrizing later
                    h_vals[j] = 0

                    adj_ind = update_adjacency_J_ij(adj_ind, i, j)

                    J_ij_vals[i,adj_ind[i]] = J_ij_vals[i, adj_ind[i]].maximum(J_ij_vals[j, adj_ind[i]])
                    J_ij_vals[adj_ind[i], i] = J_ij_vals[i,adj_ind[i]]

                    eye = chunk_deleter([j], L*L)
                    J_ij_vals = eye @ J_ij_vals @ eye
                    
                    if track_moments:
                        cluster_tracker[i] += cluster_tracker[j]
                        cluster_tracker[j] = 0
                    
                elif Omega == h_vals.max():
                    """
                    """
                    i = np.where(h_vals == h_vals.max())[0][0]
                    adj_i = adj_ind[i]
                    J_ij_new = J_ij_vals[adj_i, i] @J_ij_vals[i, adj_i]/Omega

                    J_ij_vals[adj_i, :][:, adj_i] = J_ij_new.maximum(J_ij_vals[adj_i,:][:, adj_i])

                    h_vals[i] = 0

                    eye = chunk_deleter([i], L*L)
                    J_ij_vals = eye @ J_ij_vals @ eye

                    adj_ind = update_adjacency_h(adj_ind, i)
                if step%measure_step == 0:
                    h_vals_remain = h_vals[h_vals!=0]

                    n,bins = np.histogram(-np.log(h_vals_remain), density=True, bins = 40)

                    width = bins[1]-bins[0]
                    popt, pcov = curve_fit(exponential_dist, bins[1:]-width/2, n)
                    R0_array.append(popt[1])
                    #R0_array_err.append(pcov[0][0])
                    
                    if track_moments:
                        cluster_moments = cluster_tracker[cluster_tracker!=0]
                        mu_array.append(cluster_moments.mean())
    if track_moments: return J_ij_vals, h_vals, np.array(R0_array), np.array(mu_array)
    return J_ij_vals, h_vals, np.array(R0_array)


#A, B, C = run_decimation(L, steps, a_vals, b_vals)
#print(C)
