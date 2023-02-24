#!/usr/bin/env python3
import os
import numpy as np
import time

#os.system("pwd")
for w in np.arange(5, 7, 0.2):
    os.system("nq -c mpirun run_script.py "+str(w))
    time.sleep(0.5)

