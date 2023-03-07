#!/usr/bin/env python3
import os
import numpy as np
import time

#os.system("pwd")
for L in [30]:
    for w in np.arange(0.5, 3.5, 0.2):
        for a in [0.1]:#np.arange(0.01, 0.05, 0.01):
            os.system("nq -c mpirun run_script.py "+str(w)+" "+str(L)+" "+str(a))
            time.sleep(0.5)

