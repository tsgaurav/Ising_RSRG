import numpy as np
from scipy import sparse
from sympy import Symbol, Interval
from sympy.stats import ContinuousRV, sample
from aux_funcs import *
from log_aux_funcs import *

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

def fill_beta_vals_bdry(size, bdry_dict, lambda_bdry, lambda_blk):
    # Filling the log-field coupling with exponential distribution of different variance for bulk and boundary
    bdry_size = bdry_dict.sum()
    beta_vals = np.zeros(size)
    beta_vals[bdry_dict] = np.random.exponential(scale=lambda_bdry, size=bdry_size) #Filling boundary fields
    beta_vals[~bdry_dict] = np.random.exponential(scale=lambda_blk, size=(size-bdry_size))  #Filling bulk fields
    return beta_vals

def fill_zeta_ij_bdry(size, bdry_dict, adj_ind, a_mat, b_mat, w_mat):
    #Note, here a_mat, b_mat and w_mat are 2x2 symmetric matrices giving the distribution parameters for blk-blk, blk-bdry and
    #bdry-bdry couplings
    zeta_ij_vals = sparse.lil_matrix((size, size))
    for ind in range(size):
        
        tag = int(bdry_dict[ind])
        
        #Restrict to upper triangular part of matrix
        adj_ind_array = np.array(adj_ind[ind])
        upper_ind = adj_ind_array[adj_ind_array>ind]
        
        #Seperate adjacent indices into bulk and boundary
        bdry_ind = [i for i in upper_ind if bdry_dict[i]]
        blk_ind = [i for i in upper_ind if ~bdry_dict[i]]
        
        
        zeta_ij_vals[ind, bdry_ind] = sparse.lil_matrix(np.array  \
                                    (random_lin_dist_width(a_mat[tag, 1], b_mat[tag, 1], w_mat[tag, 1], len(bdry_ind))))

        zeta_ij_vals[ind, blk_ind] = sparse.lil_matrix(np.array  \
                                    (random_lin_dist_width(a_mat[tag, 0], b_mat[tag, 0], w_mat[tag, 0], len(blk_ind))))
        
    return zeta_ij_vals + zeta_ij_vals.T
