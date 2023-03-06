#!/usr/bin/env python3
import os
import numpy as np
import time

#os.system("pwd")
for L in [50]:
    for w in [2.9]:#np.arange(1, 2.7, 0.2):
        for a in np.arange(0.01, 0.05, 0.01):
            os.system("nq -c mpirun run_script.py "+str(w)+" "+str(L)+" "+str(a))
            time.sleep(0.5)

