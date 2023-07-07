#!/usr/bin/env python3
from os import path,environ,mkdir,remove
import subprocess 
from sys import argv
import numpy as np
import time

N_nodes = 5
N_threads = 32

Run_file = """

#!/bin/bash
#PBS -l walltime=%s,select=%d:ncpus=%d:mpiprocs=%d:mem=48gb
#PBS -N RSRG
#PBS -A st-ianaffle-1
#PBS -o logs/%s_out.txt
#PBS -e errors/%s_error.txt

module load gcc
module load python
module load openmpi
module load py-virtualenv

cd /scratch/st-ianaffle-1/tenkila/Ising_RSRG
source ../env_py/bin/activate

mpiexec log_run_rect_bdry.py %s
"""

#log_run_rect_bdry.py, log_run_script_bdry.py, log_run_SO_bdry.py

for L in [80, 120, 150]:
	for w_blk in [6.3]:#np.arange(3.5,6.0, 0.2):
		for w_bdry in np.arange(0.2, 1, 0.2):
			for w_mixed in [w_blk]:#np.arange(10, 100, 20):
				w_blk = round(w_blk, 2)
				w_mixed = round(w_mixed, 2)
				w_bdry = round(w_bdry, 2)
				jobname = str(L) + str(w_blk) + str(w_bdry) + str(w_mixed)
				input_vals = str(L)+" "+str(w_blk)+" "+str(w_bdry)+" "+str(w_mixed)
				if L==40: time_request = '00:30:00'
				elif L<=80: time_request='2:00:00'
				elif L<=120: time_request='6:00:00'
				elif L<=150: time_request='10:00:00'
				ts = str(time.time())
				sfile = open(jobname+'.pbs','w')
				sfile.write(Run_file%(time_request,N_nodes, N_threads,N_threads,jobname+ts,jobname+ts,input_vals))
				sfile.close()
				subprocess.run(['qsub', jobname+'.pbs'])
				remove(jobname+'.pbs')
				time.sleep(0.5)


