#!/usr/bin/env python3
import os
import numpy as np
import time

#os.system("pwd")
for L in [20]:
    for w in [6.6]:#np.arange(6.0, 7.4, 0.05):
        for a in [0.1]:#np.arange(0.01, 0.05, 0.01):
            os.system("nq -c mpirun log_run_script.py "+str(w)+" "+str(L)+" "+str(a))
            time.sleep(0.5)

