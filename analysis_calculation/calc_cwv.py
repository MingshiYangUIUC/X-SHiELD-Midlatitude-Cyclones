"""
Author:
Mingshi Yang (mingshi3@illinois.edu)

Date:
2026-01-16

Project:
Characteristics of Midlatitude Cyclones under Climate Change in a Global Storm-Resolving Model

Description:
This script vertically integrates specific humidity (q) to compute column
water vapor. The resulting fields are used in Figure 5 of the manuscript.
"""

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import os
import gc

# input and output dirs will be the same
composite_dir = '/glade/work/mingshiy/XSHIELD/data/Composites'

experiments = [ 'PIRE',
                'PIRE_CO2_1270ppmv',
                'PIRE_PLUS_4K',
                'PIRE_PLUS_4K_CO2_1270ppmv']

var_name = 'q'
        
levels = ['1000','925','850','700','500','200','50'] # in hPa
plevels = np.array([float(l)*100 for l in levels]) # in Pa

for exp in experiments:

    print(exp)
    if os.path.isfile(f'{composite_dir}/TK2Y-{exp}_cwv.npy'):
        print('exist')
        continue
    qdata = np.stack([np.load(f'{composite_dir}/TK2Y-{exp}_q{lvl}.npy').astype(np.float32) for lvl in levels]).astype(np.float32)
    print(qdata.shape)

    qout = np.zeros(qdata.shape[1:]).astype(np.float32)
    print(qout.shape)

    pp = (np.zeros((7,qout.shape[0])) + plevels[:,None]).astype(np.float32)

    for i in tqdm(range(201)):
        for j in range(201):
            q = qdata[:,:,i,j]
            #print(q.shape,plevels.shape)
            #print(q)
            qout[:,i,j] = - np.trapz(q, pp, axis=0) / 9.81
            #print(q,pp[:,0],qout[:,i,j], np.nanmean(qout[:,i,j]))
            #quit()
    
    np.save(f'{composite_dir}/TK2Y-{exp}_cwv.npy',qout)
    del qdata, qout
    gc.collect()