#!/usr/bin/env python
from log_aux_funcs import *
#from aux_funcs import *
#import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

class log_system:
    
    def __init__(self, size, adj_ind, zeta_ij_vals, beta_vals, track_moments = False):
        
        self.size = size
        self.adj_ind = adj_ind
        self.zeta_ij_vals = zeta_ij_vals
        self.beta_vals = beta_vals
        
        self.Gamma_array = []
        self.Gamma_0 = 0
        self.Gamma = 0
        self.num_dec = 0
        
        self.clust_dict = {i:i for i in range(size)}
        self.reverse_dict = {i:[i] for i in range(size)}
        
        self.track_moments = track_moments
        self.moment_list = [1.0]


        return None
    
    def decimate(self):
        beta_min, zeta_min = self.beta_vals[self.beta_vals>0].min(), self.zeta_ij_vals.data.min()
        Gamma = min(beta_min, zeta_min)
        
        if Gamma == zeta_min: self.zeta_decimation(Gamma)
        elif Gamma == beta_min: self.beta_decimation(Gamma)
        
        self.Gamma = Gamma
        self.Gamma_array.append(Gamma)
        
        if self.track_moments: self.moment_list.append(self.get_moment())
        if False:#self.num_dec%50 == 0: 
            mask = np.any(self.zeta_ij_vals>30)

            r_ind, c_ind = mask.nonzero()
            if len(r_ind)>0:
                self.zeta_ij_vals[mask] = 0
                self.zeta_ij_vals.eliminate_zeros()

                self.adj_ind = purge_weak_bonds(self.adj_ind, r_ind, c_ind)
            
        self.num_dec+=1
        return None
    
    def zeta_decimation(self, Gamma):

        r_ind, c_ind, zeta_ij = sparse.find(self.zeta_ij_vals) 
        zeta_ind = np.where(zeta_ij == Gamma)[0][0]
        i, j = r_ind[zeta_ind], c_ind[zeta_ind]
        
        self.zeta_ij_vals.data += (self.Gamma_0 - Gamma) 
        self.beta_vals[self.beta_vals.nonzero()] += (self.Gamma_0 - Gamma) 
        
        self.clust_dict, self.reverse_dict = update_cluster(self.clust_dict, self.reverse_dict, i, j)

        self.beta_vals[i] +=  self.beta_vals[j]    
        self.beta_vals[j] = 0

        self.adj_ind = update_adjacency_zeta_ij(self.adj_ind, i, j)

        self.zeta_ij_vals[i,self.adj_ind[i]] += self.zeta_ij_vals[j, self.adj_ind[i]]
        self.zeta_ij_vals[i,self.adj_ind[i]] /= 2
        
        
        self.zeta_ij_vals[self.adj_ind[i], i] = self.zeta_ij_vals[i,self.adj_ind[i]]
        
        # Set the specified row to zero
        self.zeta_ij_vals.data[self.zeta_ij_vals.indptr[j]:self.zeta_ij_vals.indptr[j +1]] = 0

        # Set the specified column to zero
        bool_arr = self.zeta_ij_vals.indices == j
        self.zeta_ij_vals.data[bool_arr] = 0
        
        self.zeta_ij_vals.eliminate_zeros()
        
        return None
    
    def beta_decimation(self, Gamma):
        i = np.where(self.beta_vals == Gamma)[0][0]
        adj_i = self.adj_ind[i]
        
        self.zeta_ij_vals.data += (self.Gamma_0 - Gamma) 
        self.beta_vals[self.beta_vals.nonzero()] += (self.Gamma_0 - Gamma) 
        
        """
        zeta_subblock = self.zeta_ij_vals[np.ix_(adj_i, adj_i)].toarray()
        old_couplings = sparse.find(self.zeta_ij_vals[adj_i,i])[2]
        new_couplings = np.add.outer(old_couplings, old_couplings)
        np.fill_diagonal(new_couplings, 0)
        zeta_subblock[np.where(zeta_subblock==0)]=1000
        new_couplings = np.minimum(zeta_subblock, new_couplings)
        self.zeta_ij_vals[np.ix_(adj_i, adj_i)] = new_couplings
        
        """
        ### GPTest
        
        zeta_subblock = self.zeta_ij_vals[adj_i, :][:, adj_i].toarray()

        old_couplings = self.zeta_ij_vals[adj_i, i].data

        new_couplings = np.add.outer(old_couplings, old_couplings)
        np.fill_diagonal(new_couplings, 0)

        zeta_subblock[np.where(zeta_subblock == 0)] = 1000
        
        np.minimum(zeta_subblock, new_couplings, out=zeta_subblock)

        self.zeta_ij_vals[np.ix_(adj_i, adj_i)] = sparse.csr_matrix(zeta_subblock)
        
        # Set the specified row to zero
        self.zeta_ij_vals.data[self.zeta_ij_vals.indptr[i]:self.zeta_ij_vals.indptr[i +1]] = 0

        # Set the specified column to zero
        bool_arr = self.zeta_ij_vals.indices == i
        self.zeta_ij_vals.data[bool_arr] = 0
        
        self.zeta_ij_vals.eliminate_zeros()
        
        self.adj_ind = update_adjacency_beta(self.adj_ind, i)
        
        return None
        
        
    def get_moment(self):
        rd = self.reverse_dict
        clust_size_list = np.array([len(clust) for clust in rd.values() if clust is not None and self.beta_vals[clust].any()>0])
        return clust_size_list.mean()
