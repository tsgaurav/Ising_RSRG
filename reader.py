#!/usr/bin/env python
from aux_funcs import *
from log_aux_funcs import*
from bdry_log_aux_funcs import *

def output_reader(measure_list, fname):
    J_dist_dict = [{} for _ in range(len(measure_list))]
    h_dist_dict = [{} for _ in range(len(measure_list))]

    #In1629_J_m3571
    C = 0
    with open(fname, 'r') as reader:
        next(reader)
        for line in reader:
            marker_end = line.find('[')
            inst_num = int(line[2:6])
            coupling_type = line[7:8]
            measure_ind = np.where(measure_list==int(line[10:marker_end]))[0][0]
            try:
                couplings = np.array(json.loads(line[marker_end:]))
            except:
                print("Corrupted file")
            if coupling_type == 'h':
                h_dist_dict[measure_ind][inst_num] = couplings
            elif coupling_type == 'J':
                J_dist_dict[measure_ind][inst_num] = couplings
    
    return [h_dist_dict, J_dist_dict]

def unpack_dictionaries(file_out, measure_list):
    h_dist_dict, J_dist_dict = file_out
    h_dist_list, J_dist_list = [], []
    
    for i in range(len(measure_list)):
        h_dist_list.append(np.concatenate(tuple(h_dist_dict[i].values())))
        J_dist_list.append(np.concatenate(tuple(J_dist_dict[i].values())))
        
    return h_dist_list, J_dist_list


def output_reader_bdry(measure_list, fname):
    hblk_dist_dict = [{} for _ in range(len(measure_list))]
    hbdry_dist_dict = [{} for _ in range(len(measure_list))]
    bdry_dict_comp_dict = [{} for _ in range(len(measure_list))]
    reverse_dict_comp_dict = [{} for _ in range(len(measure_list))]
    J_blk_blk_dict = [{} for _ in range(len(measure_list))]
    J_bdry_bdry_dict = [{} for _ in range(len(measure_list))]
    J_blk_bdry_dict = [{} for _ in range(len(measure_list))]

    #In2029_hbd_m19

    with open(fname, 'r') as reader:
        next(reader)
        for line in reader:
            #marker_end = line.find('[')
            inst_num = int(line[2:6])
            coupling_type = line[7:10]
            measure_ind = int(line[12:14])
            couplings = np.array(json.loads(line[14:]))
            
            if coupling_type=='hbl':
                hblk_dist_dict[measure_ind][inst_num] = couplings
            elif coupling_type=='hbd':
                hbdry_dist_dict[measure_ind][inst_num] = couplings
            elif coupling_type== 'crd':
                reverse_dict_comp_dict[measure_ind][inst_num] = couplings
            elif coupling_type== 'cbd':
                bdry_dict_comp_dict[measure_ind][inst_num] = couplings
            elif coupling_type== 'Jll':
                J_blk_blk_dict[measure_ind][inst_num] = couplings
            elif coupling_type== 'Jrr':
                J_bdry_bdry_dict[measure_ind][inst_num] = couplings
            elif coupling_type== 'Jlr':
                J_blk_bdry_dict[measure_ind][inst_num] = couplings
    return [hblk_dist_dict, hbdry_dist_dict, bdry_dict_comp_dict, reverse_dict_comp_dict, J_blk_blk_dict, J_bdry_bdry_dict, J_blk_bdry_dict]

    