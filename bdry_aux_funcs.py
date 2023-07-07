import numpy as np
from scipy import sparse
from scipy.sparse import dok_matrix
from sympy import Symbol, Interval
from sympy.stats import ContinuousRV, sample
from aux_funcs import *

#Returns the the adjacency dictionary for an LxL triangle lattice with open boundaries along 
#the x-direction (i.e. x=0 and x=L-1 correspond to boundaries of the system)

def triangle_lattice_boundary_dictionary(L, include_nnn=False):
    ind_dict = {}
    adj_ind = {}
    bdry_dict = np.zeros(L*L,dtype=bool)
    
    for x in range(L):
        for y in range(L):
            ind_0 = index_map(x,y,L)
            
            #nn indices
            adjs = [index_map(x,(y-1)%L, L), index_map(x,(y+1)%L, L)]
            
            if include_nnn: adjs += [index_map(x,(y+2)%L, L), index_map(x,(y-2)%L, L)]
            
            if x-1>=0:
                adjs.append(index_map((x-1)%L,y, L))
                adjs.append(index_map((x-1)%L,(y-(-1)**(x%2))%L, L))
                
                if include_nnn: adjs += [index_map((x-1)%L,(y+(-1)**(x%2))%L, L), index_map((x-1)%L,(y-2*(-1)**(x%2))%L, L)]
            if x+1<L:
                adjs.append(index_map((x+1),y, L))
                adjs.append(index_map((x+1),(y-(-1)**(x%2))%L, L))
                
                if include_nnn: adjs += [index_map((x+1)%L,(y+(-1)**(x%2))%L, L), index_map((x+1)%L,(y-2*(-1)**(x%2))%L, L)]
            
            if include_nnn:
                if x-2>=0:
                    adjs += [index_map((x-2)%L,y, L), index_map((x-2)%L,(y+1)%L, L), index_map((x-2)%L,(y-1)%L, L)]
                if x+2<L:
                    adjs += [index_map((x+2)%L,y, L), index_map((x+2)%L,(y+1)%L, L), index_map((x+2)%L,(y-1)%L, L)]
            
            if x==0: bdry_dict[ind_0] = True
            elif x==(L-1): bdry_dict[ind_0] = True      
                
            ind_dict[ind_0] = (x,y)
            adj_ind[ind_0] = list(set(adjs))
    return ind_dict, adj_ind, bdry_dict

def fill_h_vals_bdry(size, bdry_dict, lambda_bdry, lambda_blk):
    # Filling the log-field coupling with exponential distribution of different variance for bulk and boundary
    bdry_size = bdry_dict.sum()
    h_vals = np.zeros(size)
    h_vals[bdry_dict] = np.exp(-np.random.exponential(scale=lambda_bdry, size=bdry_size)) #Filling boundary fields
    h_vals[~bdry_dict] = np.exp(-np.random.exponential(scale=lambda_blk, size=(size-bdry_size)))  #Filling bulk fields
    return h_vals

def fill_J_ij_bdry(size, bdry_dict, adj_ind, a_mat, b_mat, w_mat):
    #Note, here a_mat, b_mat and w_mat are 2x2 symmetric matrices giving the distribution parameters for blk-blk, blk-bdry and
    #bdry-bdry couplings
    J_ij_vals = sparse.lil_matrix((size, size))
    for ind in range(size):
        
        tag = int(bdry_dict[ind])
        
        #Restrict to upper triangular part of matrix
        adj_ind_array = np.array(adj_ind[ind])
        upper_ind = adj_ind_array[adj_ind_array>ind]
        
        #Seperate adjacent indices into bulk and boundary
        bdry_ind = [i for i in upper_ind if bdry_dict[i]]
        blk_ind = [i for i in upper_ind if ~bdry_dict[i]]
        
        
        J_ij_vals[ind, bdry_ind] = sparse.lil_matrix(np.exp(-np.array  \
                                    (random_lin_dist_width_v2(a_mat[tag, 1], b_mat[tag, 1], w_mat[tag, 1], len(bdry_ind)))))

        J_ij_vals[ind, blk_ind] = sparse.lil_matrix(np.exp(-np.array  \
                                    (random_lin_dist_width_v2(a_mat[tag, 0], b_mat[tag, 0], w_mat[tag, 0], len(blk_ind)))))
        
    return J_ij_vals + J_ij_vals.T

class RandomLinDistWidthSampler:
    def __init__(self, a, b, w, L):
        self.L = L
        X = Symbol('x')
        self.distributions = [ContinuousRV(X, (a[0,0]+b[0,0]*X)/(a[0,0]*w[0,0] + b[0,0]*w[0,0]**2/2), Interval(0, w[0,0])),\
                              ContinuousRV(X, (a[0,1]+b[0,1]*X)/(a[0,1]*w[0,1] + b[0,1]*w[0,1]**2/2), Interval(0, w[0,1])),\
                              ContinuousRV(X, (a[1,1]+b[1,1]*X)/(a[1,1]*w[1,1] + b[1,1]*w[1,1]**2/2), Interval(0, w[1,1]))]
        self.samples_cache = [[None, None], [None, None]]

    def sample(self, coupling_type, n_samples):
        i = coupling_type // 2
        j = coupling_type % 2
        if self.samples_cache[i][j] is None or len(self.samples_cache[i][j]) < n_samples:
            self.samples_cache[i][j] = sample(self.distributions[coupling_type], size=(self.L*n_samples))  # get 10x needed
        result = self.samples_cache[i][j][:n_samples]
        self.samples_cache[i][j] = self.samples_cache[i][j][n_samples:]
        return result

def fill_zeta_ij_bdry_v2(size, bdry_dict, sampler):
    zeta_ij_vals = dok_matrix((size, size))
    
    for ind in range(size):
        tag = int(bdry_dict[ind])
        
        adj_ind_array = np.array(adj_ind[ind])
        upper_ind = adj_ind_array[adj_ind_array>ind]
        
        bdry_ind = upper_ind[bdry_dict[upper_ind]]
        blk_ind = upper_ind[~bdry_dict[upper_ind]]
        
        zeta_ij_vals[ind, bdry_ind] = sampler.sample(tag+1, len(bdry_ind))
        zeta_ij_vals[ind, blk_ind] = sampler.sample(tag, len(blk_ind))
    
    zeta_ij_vals = zeta_ij_vals.tocsr()
    
    return zeta_ij_vals + zeta_ij_vals.T
