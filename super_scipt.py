#!/usr/bin/env python3
import os
import numpy as np
import time

#os.system("pwd")
for w in [4.7, 4.9]:#np.arange(3, 5, 0.2):
    os.system("nq -c mpirun run_script.py "+str(w))
    time.sleep(0.5)

