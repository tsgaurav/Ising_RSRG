#!/usr/bin/env python3
import os
import numpy as np
import time

#os.system("pwd")
for L in [50]:
    for w in [1.8, 2.0, 2.2]:#np.arange(2.4, 2.8, 0.02):
        for a in [0.1]:#np.arange(0.01, 0.05, 0.01):
            os.system("nq -c mpirun run_script.py "+str(w)+" "+str(L)+" "+str(a))
            time.sleep(0.5)

