#!/usr/bin/env python3
from os import path,environ,mkdir,remove
import subprocess 
from sys import argv
import numpy as np
import time

N_threads = 32

Run_file = """

#!/bin/bash
#PBS -l walltime=%s,select=1:ncpus=%d:mpiprocs=%d:mem=32gb
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

mpiexec run_script_bdry.py %s


"""

for L in [80, 100]:
	for w_blk in np.arange(2.4,3.2, 0.04):
		for w_bdry in [0.1]:#np.arange(0.01, 0.05, 0.01):
			for w_mixed in [2.6]:
				w_mixed = w_blk
				w_bdry = w_blk
				jobname = str(L) + str(w_blk) + str(w_bdry) + str(w_mixed)
				input_vals = str(L)+" "+str(w_blk)+" "+str(w_bdry)+" "+str(w_mixed)
				time_request='10:00:00'
				ts = str(time.time())
				sfile = open(jobname+'.pbs','w')
				sfile.write(Run_file%(time_request,N_threads,N_threads,jobname+ts,jobname+ts,input_vals))
				sfile.close()
				subprocess.run(['qsub', jobname+'.pbs'])
				remove(jobname+'.pbs')
				time.sleep(0.5)


