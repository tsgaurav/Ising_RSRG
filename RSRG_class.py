#!/usr/bin/env python
from aux_funcs import *
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from iminuit import cost, Minuit


class system:
    
    def __init__(self, size, adj_ind, J_ij_vals, h_vals, sparsify=False):
        
        self.size = size
        self.adj_ind = adj_ind
        self.J_ij_vals = J_ij_vals
        self.h_vals = h_vals
        self.sparsify = sparsify
        
        self.N = 0
        self.R0_array = []
        self.Gamma_array = []
        self.Omega_array = []
        self.coupling_dec_list = []
        self.Omega_0 = max(h_vals.max(), J_ij_vals.max())
        self.Omega = self.Omega_0 
        
        self.clust_dict = {i:i for i in range(size)}
        self.reverse_dict = {i:[i] for i in range(size)}  #Cluster index is key and vals is list with lattice indices in cluster 
        
        
        return None
    
    def decimate(self):
        self.N += 1
        Omega = max(self.h_vals.max(), self.J_ij_vals.max())
        self.Omega = Omega
        self.Gamma_array.append(np.log(self.Omega_0/Omega))
        self.Omega_array.append(Omega)
        if Omega == self.J_ij_vals.max(): 
            self.J_decimation(Omega)
            self.coupling_dec_list.append(True)
        elif Omega == self.h_vals.max(): 
            self.h_decimation(Omega)
            self.coupling_dec_list.append(False)
         
        #if self.N%self.measure_step==0: self.R0_array.append(self.extract_width())
        
        return None
    
    def J_decimation(self, Omega):
        r_ind, c_ind, J_ij = sparse.find(self.J_ij_vals) 
        J_ind = np.where(J_ij == Omega)[0][0]
        i, j = r_ind[J_ind], c_ind[J_ind]

        self.h_vals[i] = self.h_vals[i]*self.h_vals[j]/Omega    
        self.h_vals[j] = 0

        update_adjacency_J_ij(self.adj_ind, i, j)
        
        self.clust_dict, self.reverse_dict = update_cluster(self.clust_dict, self.reverse_dict, i, j)

        self.J_ij_vals[i,self.adj_ind[i]] = self.J_ij_vals[i, self.adj_ind[i]].maximum(self.J_ij_vals[j, self.adj_ind[i]])
        self.J_ij_vals[self.adj_ind[i], i] = self.J_ij_vals[i,self.adj_ind[i]]

        eye = chunk_deleter([j], self.size)
        self.J_ij_vals = eye @ self.J_ij_vals @ eye
        return None
    
    def h_decimation(self, Omega):
        i = np.where(self.h_vals == self.h_vals.max())[0][0]
        adj_i = self.adj_ind[i]
        J_ij_new = self.J_ij_vals[adj_i, i] @self.J_ij_vals[i, adj_i]/Omega
        
        if self.sparsify: J_ij_new.data[J_ij_new.data<self.Omega_0/2]=0
        
        self.J_ij_vals[adj_i, :][:, adj_i] = J_ij_new.maximum(self.J_ij_vals[adj_i,:][:, adj_i])
        
        self.h_vals[i] = 0

        eye = chunk_deleter([i], self.size)

        self.J_ij_vals = eye @ self.J_ij_vals @ eye

        update_adjacency_h(self.adj_ind, i)
        return None
    
    def extract_width(self):
        #Get width of exponential distribution by fitting field couplings to normalized exponential
        h_remain = self.h_vals[self.h_vals!=0]
        c = cost.UnbinnedNLL(-np.log(h_remain/self.Omega), exponential_dist_norm)
        m = Minuit(c, a=0.8)
        m.migrad()
        return m.values[0]
    
class log_system:
    
    def __init__(self, size, adj_ind, zeta_ij_vals, beta_vals):
        
        self.size = size
        self.adj_ind = adj_ind
        self.zeta_ij_vals = zeta_ij_vals
        self.beta_vals = beta_vals
        
        self.Gamma_array = []
        self.Gamma_0 = min(beta_vals.min(), sparse.find(zeta_ij_vals)[2].min())
        self.Gamma = self.Gamma_0
        
        self.N = 0
        self.R0_array = []
        self.coupling_dec_list = []

        
        self.clust_dict = {i:i for i in range(size)}
        self.reverse_dict = {i:[i] for i in range(size)}  #Cluster index is key and vals is list with lattice indices in cluster 
        
        return None
    
    def decimate(self):
        beta_min, zeta_min = self.beta_vals[self.beta_vals>0].min(), sparse.find(self.zeta_ij_vals)[2].min()
        Gamma = min(beta_min, zeta_min)
        self.Gamma = Gamma
        self.Gamma_array.append(Gamma)
        if Gamma == zeta_min: 
            self.zeta_decimation(Gamma)
            self.coupling_dec_list.append(True)
        elif Gamma == beta_min: 
            self.beta_decimation(Gamma)
            self.coupling_dec_list.append(False)
        
        return None
    
    def zeta_decimation(self, Gamma):
        r_ind, c_ind, zeta_ij = sparse.find(self.zeta_ij_vals) 
        zeta_ind = np.where(zeta_ij == Gamma)[0][0]
        i, j = r_ind[zeta_ind], c_ind[zeta_ind]
        
        self.clust_dict, self.reverse_dict = update_cluster(self.clust_dict, self.reverse_dict, i, j)

        self.beta_vals[i] = self.beta_vals[i] + self.beta_vals[j]    
        self.beta_vals[j] = 0

        update_adjacency_zeta_ij(self.adj_ind, i, j)

        self.zeta_ij_vals[i,self.adj_ind[i]] = \
        (self.zeta_ij_vals[i, self.adj_ind[i]] + self.zeta_ij_vals[j, self.adj_ind[i]])/2
        
        
        self.zeta_ij_vals[self.adj_ind[i], i] = self.zeta_ij_vals[i,self.adj_ind[i]]

        eye = chunk_deleter([j], self.size)
        self.zeta_ij_vals = eye @ self.zeta_ij_vals @ eye
        return None
    
    def beta_decimation(self, Gamma):
        i = np.where(self.beta_vals == Gamma)[0][0]
        adj_i = self.adj_ind[i]

        #zeta_ij_new = np.outer(self.zeta_ij_vals[adj_i, i].toarray(),np.ones(len(adj_i))) 
        #zeta_ij_new = zeta_ij_new + zeta_ij_new.T
        
        zeta_ij_new = np.add.outer(self.zeta_ij_vals[adj_i, i].toarray()[:,0], self.zeta_ij_vals[adj_i, i].toarray()[:,0])

        self.zeta_ij_vals[adj_i, :][:, adj_i] = (zeta_ij_new + self.zeta_ij_vals[adj_i,:][:, adj_i])/2

        self.beta_vals[i] = 0

        eye = chunk_deleter([i], self.size)
        self.zeta_ij_vals = eye @ self.zeta_ij_vals @ eye

        update_adjacency_beta(self.adj_ind, i)
        return None
        
