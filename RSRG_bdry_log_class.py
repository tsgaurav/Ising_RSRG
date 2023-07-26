#!/usr/bin/env python
from aux_funcs import *
from log_aux_funcs import*
from bdry_log_aux_funcs import *

class bdry_log_system:
    
    def __init__(self, size, adj_ind, bdry_dict, zeta_ij_vals, beta_vals, track_moments = False):
        
        self.size = size
        self.adj_ind = adj_ind
        self.zeta_ij_vals = zeta_ij_vals
        self.beta_vals = beta_vals
        self.bdry_dict = bdry_dict
        
        self.Gamma_array = []
        self.Gamma_0 = 0
        self.Gamma = 0
        self.num_dec = 0
        
        self.clust_dict = {i:i for i in range(size)}
        self.reverse_dict = {i:[i] for i in range(size)}
        
        self.track_moments = track_moments
        self.bdry_moment_list = [1.0]
        self.blk_moment_list = [1.0]
        self.active_clust_list = np.ones(size, dtype=bool)

        return None
    
    def decimate(self):
        beta_min, zeta_min = self.beta_vals[self.beta_vals>0].min(), self.zeta_ij_vals.data.min()
        Gamma = min(beta_min, zeta_min)
        
        if Gamma == zeta_min: self.zeta_decimation(Gamma)
        elif Gamma == beta_min: self.beta_decimation(Gamma)
        
        self.Gamma = Gamma
        self.Gamma_array.append(Gamma)
        
        if self.track_moments: 
            self.blk_moment_list.append(self.get_moment_bulk())
            self.bdry_moment_list.append(self.get_moment_bdry())
        if False:#self.num_dec%50 == 0: 
            mask = np.any(self.zeta_ij_vals>10)

            r_ind, c_ind = mask.nonzero()
            if len(r_ind)>0:
                self.zeta_ij_vals[mask] = 0
                self.zeta_ij_vals.eliminate_zeros()

                self.adj_ind = purge_weak_bonds(self.adj_ind, r_ind, c_ind)
            
        self.num_dec+=1
        
        beta_remain_bdry = self.beta_vals[self.bdry_dict]
        if len(np.nonzero(beta_remain_bdry)[0])==1:
            site = np.where(beta_remain_bdry[np.nonzero(beta_remain_bdry)[0][0]]==self.beta_vals)[0][0]
            self.final_bdry_clust = (site, sum(self.Gamma_array), self.beta_vals[site])
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
        
        self.bdry_dict[i] = self.bdry_dict[i] or self.bdry_dict[j]
        self.bdry_dict[j] = self.bdry_dict[i]

        self.adj_ind = update_adjacency_zeta_ij(self.adj_ind, i, j)

        subblock = self.zeta_ij_vals[i, self.adj_ind[i]].toarray()
        subblock[np.where(subblock == 0)] = 1000
        subblock_old = self.zeta_ij_vals[j, self.adj_ind[i]].toarray()
        subblock_old[np.where(subblock_old == 0)] = 1000
        subblock = np.minimum(subblock, subblock_old)
        subblock[np.where(subblock==1000)] = 0
        self.zeta_ij_vals[i, self.adj_ind[i]] = sparse.csr_matrix(subblock)
        self.zeta_ij_vals[self.adj_ind[i], i] = self.zeta_ij_vals[i,self.adj_ind[i]]
        
        
        self.zeta_ij_vals[self.adj_ind[i], i] = self.zeta_ij_vals[i,self.adj_ind[i]]
        
        # Set the specified row to zero
        self.zeta_ij_vals.data[self.zeta_ij_vals.indptr[j]:self.zeta_ij_vals.indptr[j +1]] = 0

        # Set the specified column to zero
        bool_arr = self.zeta_ij_vals.indices == j
        self.zeta_ij_vals.data[bool_arr] = 0
        
        self.zeta_ij_vals.eliminate_zeros()
        
        self.active_clust_list[j] = False

        return None
    
    def beta_decimation(self, Gamma):

        i = np.flatnonzero(self.beta_vals == Gamma)[0]
        adj_i = self.adj_ind[i]


        self.zeta_ij_vals.data += (self.Gamma_0 - Gamma)
        self.beta_vals[self.beta_vals.nonzero()] += (self.Gamma_0 - Gamma)

        # Retrieve subblock as dense matrix and perform computations in-place
        zeta_subblock = self.zeta_ij_vals[adj_i, :][:, adj_i].toarray()

        old_couplings = self.zeta_ij_vals[adj_i, i].data
        new_couplings = np.add.outer(old_couplings, old_couplings)
        np.fill_diagonal(new_couplings, 0)

        # Update zeta_subblock in-place
        zeta_subblock[zeta_subblock == 0] = 1000
        np.minimum(zeta_subblock, new_couplings, out=zeta_subblock)

        # Convert updated zeta_subblock back to sparse and update zeta_ij_vals in-place
        self.zeta_ij_vals[np.ix_(adj_i, adj_i)] = sparse.csr_matrix(zeta_subblock)

        # Set the specified row to zero
        self.zeta_ij_vals.data[self.zeta_ij_vals.indptr[i]:self.zeta_ij_vals.indptr[i +1]] = 0

        # Set the specified column to zero
        self.zeta_ij_vals.data[self.zeta_ij_vals.indices == i] = 0

        # Remove zero entries from the sparse matrix
        self.zeta_ij_vals.eliminate_zeros()

        # Update adjacency
        self.adj_ind = update_adjacency_beta(self.adj_ind, i)
        
        self.active_clust_list[i] = False

        return None
        
        
    def get_moment_bdry(self):
        rd = self.reverse_dict
        clust_size_list = np.array([len(clust) for clust in rd.values() if clust is not None and self.beta_vals[clust].any()>0 and self.bdry_dict[clust[0]]])
        return clust_size_list.mean()
    
    def get_moment_bulk(self):
        rd = self.reverse_dict
        clust_size_list = np.array([len(clust) for clust in rd.values() if clust is not None and self.beta_vals[clust].any()>0 and ~self.bdry_dict[clust[0]]])
        return clust_size_list.mean()

