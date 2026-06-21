"""
Author:
Mingshi Yang (mingshi3@illinois.edu)

Date:
2026-01-16

Project:
Characteristics of Midlatitude Cyclones under Climate Change in a Global Storm-Resolving Model

Description:
This script vertically integrates specific humidity (q) to compute column
water vapor.

Update History:
    2026-06-21
        - Added configurable upper and lower boundaries of vertical integration.
        - Added support to integrate q tendency to get column water tendency.
        - Improve speed by using more efficient memory maps.

"""

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import os
import gc

# input and output dirs will be the same
composite_dir = '/data/keeling/a/mingshi3/c/xshield/composites'

experiments = [ 'PIRE',
                'PIRE_CO2_1270ppmv',
                'PIRE_PLUS_4K',
                'PIRE_PLUS_4K_CO2_1270ppmv']

var_name = 'dqdt'
        
levels = ['1000','925','850','700','500','200','50'] # in hPa

lower_bound = '925'
upper_bound = '200'

# select levels within bound
ilevel0 = levels.index(lower_bound)
ilevel1 = levels.index(upper_bound)
levels = levels[ilevel0:ilevel1+1]

print('Using levels', levels)


plevels = np.array([float(l)*100 for l in levels]) # in Pa

for exp in experiments:

    print(exp)

    if var_name == 'q':
        file_out = f'{composite_dir}/TK2Y-{exp}_cwv_{lower_bound}-{upper_bound}.npy'
    else:
        file_out = f'{composite_dir}/TK2Y-{exp}_c{var_name}_{lower_bound}-{upper_bound}.npy'

    if os.path.isfile(file_out):
        print('exist')
        continue

    # open level files as read-only memmaps
    qfiles = [
        np.load(
            f'{composite_dir}/TK2Y-{exp}_{var_name}{lvl}.npy',
            mmap_mode='r'
        )
        for lvl in levels
    ]

    nt, ny, nx = qfiles[0].shape
    print((len(qfiles), nt, ny, nx))

    # keep output in RAM
    qout = np.empty((nt, ny, nx), dtype=np.float32)

    pp = plevels.astype(np.float32)

    for i in tqdm(range(nt)):
        # shape: nlevel, ny, nx
        q_slice = np.stack([
            qf[i, :, :]
            for qf in qfiles
        ], axis=0).astype(np.float32)

        qout[i, :, :] = -np.trapz(q_slice, pp[:, None, None], axis=0) / 9.81

    np.save(file_out, qout)

    del qfiles, qout
    gc.collect()