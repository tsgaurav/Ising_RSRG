#!/usr/bin/env python3
import os
import numpy as np
import time

#os.system("pwd")
for L in [50]:
    for w in [4.5]:#np.arange(3.5, 6, 0.2):
        for a in [0.1]:#np.arange(0.1, 0.14, 0.02):
            os.system("nq -c mpirun run_script.py "+str(w)+" "+str(L)+" "+str(a))
            time.sleep(0.5)

