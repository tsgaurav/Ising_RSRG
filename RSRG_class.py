#!/usr/bin/env python
from aux_funcs import *
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from iminuit import cost, Minuit


class system:
    
    def __init__(self, size, adj_ind, J_ij_vals, h_vals, measure_step=20):
        
        self.size = size
        self.adj_ind = adj_ind
        self.J_ij_vals = J_ij_vals
        self.h_vals = h_vals
        
        self.measure_step = measure_step
        self.N = 0
        self.R0_array = []
        self.Gamma_array = []
        self.Omega_0 = max(h_vals.max(), J_ij_vals.max())
        self.Omega = self.Omega_0 
        return None
    
    def decimate(self):
        self.N += 1
        Omega = max(self.h_vals.max(), self.J_ij_vals.max())
        self.Omega = Omega
        self.Gamma_array.append(np.log(self.Omega_0/Omega))
        if Omega == self.J_ij_vals.max(): self.J_decimation(Omega)
        elif Omega == self.h_vals.max(): self.h_decimation(Omega)
         
        if self.N%self.measure_step==0: self.R0_array.append(self.extract_width())
        
        return None
    
    def J_decimation(self, Omega):
        r_ind, c_ind, J_ij = sparse.find(self.J_ij_vals) 
        J_ind = np.where(J_ij == Omega)[0][0]
        i, j = r_ind[J_ind], c_ind[J_ind]

        self.h_vals[i] = self.h_vals[i]*self.h_vals[j]/Omega    
        self.h_vals[j] = 0

        update_adjacency_J_ij(self.adj_ind, i, j)

        self.J_ij_vals[i,self.adj_ind[i]] = self.J_ij_vals[i, self.adj_ind[i]].maximum(self.J_ij_vals[j, self.adj_ind[i]])
        self.J_ij_vals[self.adj_ind[i], i] = self.J_ij_vals[i,self.adj_ind[i]]

        eye = chunk_deleter([j], self.size)
        self.J_ij_vals = eye @ self.J_ij_vals @ eye
        return None
    
    def h_decimation(self, Omega):
        i = np.where(self.h_vals == self.h_vals.max())[0][0]
        adj_i = self.adj_ind[i]
        J_ij_new = self.J_ij_vals[adj_i, i] @self.J_ij_vals[i, adj_i]/Omega

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
