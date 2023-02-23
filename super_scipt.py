#!/usr/bin/env python3
import os
import numpy as np

#os.system("pwd")
for w in np.arange(3, 7, 0.5):
    os.system("nq -q mpirun -q 6 run_script.py "+str(w))
