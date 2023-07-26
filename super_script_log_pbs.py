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
#PBS -l walltime=%s,select=%d:ncpus=%d:mpiprocs=%d:mem=64gb
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

mpiexec log_run_script.py %s
"""

#os.system("pwd")
for L in [10, 20, 40, 80]:
	for w in np.arange(5.5, 8, 0.2):
		w = round(w, 2)
		for a in [0.1]:#np.arange(0.01, 0.2, 0.01):
			jobname = str(L)+str(w)+str(a)
			input_vals = str(w)+" "+str(L)+" "+str(a)
			if L<=40: time_request='00:30:00'
			elif L==50: time_request='01:00:00'
			elif L<=64: time_request='02:00:00'
			elif L==80: time_request='10:00:00'
			elif L==100: time_request='16:00:00'
			#time_request = '36:00:00'
			ts = str(time.time())
			sfile = open(jobname+'.pbs','w')
			sfile.write(Run_file%(time_request,N_nodes, N_threads,N_threads,jobname+ts,jobname+ts,input_vals))
			sfile.close()	    
			subprocess.run(['qsub', jobname+'.pbs'])
			remove(jobname+'.pbs')
			time.sleep(0.5)
            

