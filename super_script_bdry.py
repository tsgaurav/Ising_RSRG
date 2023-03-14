#!/usr/bin/env python3
import os
import numpy as np
import time

#os.system("pwd")
for L in [30, 50, 70]:
    for w_blk in np.arange(1.4, 4.0, 0.2):
        for w_bdry in [0.1]:#np.arange(0.01, 0.05, 0.01):
            for w_mixed in [2.6]:
                os.system("nq -c mpirun run_script_bdry.py "+str(L)+" "+str(w_blk)+" "+str(w_bdry)+" "+str(w_mixed))
                time.sleep(0.5)

