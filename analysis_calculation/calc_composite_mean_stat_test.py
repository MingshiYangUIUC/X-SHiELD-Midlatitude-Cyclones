"""
Author:
Mingshi Yang (mingshi3@illinois.edu)

Date:
2026-01-16

Project:
Characteristics of Midlatitude Cyclones under Climate Change in a Global Storm-Resolving Model

Description:
This script computes composite mean fields from all available composite samples
and applies statistical significance testing (such as t-test) between
sensitivity experiments and the control (CTRL). The workflow is applied
systematically across all experiments, user-specified variables, and vertical
levels defined by the user.
"""

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
from tqdm import tqdm
from scipy.stats import ttest_ind, mannwhitneyu # or other tests

# input and output directory
composite_dir = '/glade/work/mingshiy/XSHIELD/data/Composites'
composite_derived_dir = '/glade/work/mingshiy/XSHIELD/data/Composites_derived'


experiments = [ 'PIRE',
                'PIRE_CO2_1270ppmv',
                'PIRE_PLUS_4K',
                'PIRE_PLUS_4K_CO2_1270ppmv']

# (a) for derived variables, omit levels, use custom variable name
var_name = 'stability-thetae-200925'
#var_name = 'PRATEsfc'
var_name = 'uv925'
levels = ['']

# (b) or define regular variable names and levels, 
#var_name = 'rh'
#levels = [1000,925,850,700,500,200,50]

# import and define statistical test function
test = ttest_ind

tk = []
idx = []
for exp in experiments: # load track and indices
    tk.append(np.loadtxt(f'/glade/work/mingshiy/XSHIELD/tracks/TK2Y-{exp}/All_Tracks.txt'))
    idx.append(np.loadtxt(f'/glade/work/mingshiy/XSHIELD/tracks/TK2Y-{exp}/All_Indexes.txt'))

Y, X = 201,201

for level in levels:

    mean_data = []
    p_data = []

    for i,exp in enumerate(experiments):
        print(level, exp)

        if i == 0:
            # CTRL experiment data: only create composite mean
            d = tk[i]
            sel0 = np.array((d[:,0]>2020000000)&(d[:,0]<2022000000)&(d[:,1]<65)&(d[:,1]>=30)) # subset, because all track include full simulation period and other latitudes
            data_ctrl = np.load(f'{composite_dir}/TK2Y-{exp}_{var_name}{level}.npy')[sel0].astype(np.float32)
            print(data_ctrl.shape)
            mean_data.append(np.nanmean(data_ctrl,axis=0))
        else:
            # sensitivity experiments: get composite mean and conduct statistical test
            d = tk[i]
            sel = np.array((d[:,0]>2020000000)&(d[:,0]<2022000000)&(d[:,1]<65)&(d[:,1]>=30))
            data_test = np.load(f'{composite_dir}/TK2Y-{exp}_{var_name}{level}.npy')[sel].astype(np.float32)
            mean_data.append(np.nanmean(data_test,axis=0))
            
            # loop over grid for stat test
            #t_stats = np.zeros((Y, X))
            p_values = np.zeros((Y, X))
            for i in tqdm(range(Y)):
                for j in range(X):
                    t_stat, p_value = test(data_ctrl[:, i, j], data_test[:, i, j],nan_policy='omit', equal_var=False)
                    #t_stats[i, j] = t_stat
                    p_values[i, j] = p_value
                    if np.isnan(p_value):
                        print(i,j)

            p_data.append(p_values)
            del data_test
    del data_ctrl

    np.save(f'{composite_derived_dir}/TK2Y-allexp_means_{var_name}{level}_t.npy',np.array(mean_data))
    np.save(f'{composite_derived_dir}/TK2Y-allexp-ctrl_pvalue_{var_name}{level}_t.npy',np.array(p_data))
