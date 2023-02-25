#!/usr/bin/env python3
import os
import numpy as np
import time

#os.system("pwd")
for L in [100, 200]:
    for w in np.arange(3.5, 6, 0.2):

        os.system("nq -c mpirun run_script.py "+str(w)+" "+str(L))
        time.sleep(0.5)

