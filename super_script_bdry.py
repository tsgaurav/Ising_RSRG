#!/usr/bin/env python3
import os
import numpy as np
import time

#os.system("pwd")
for Lx in [10]:
    for L in [15]:
        for w in np.arange(2.0, 3.0, 0.15):
            w_bdry = round(w, 3)
            w_blk, w_mixed = 6.6, 6.6

            os.system("nq -c mpirun log_run_rect_bdry.py "+str(L)+" "+str(w_blk)+" "+str(w_bdry)+" "+str(w_mixed)+" "+str(Lx))
            time.sleep(0.5)

