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

Update History:
    2026-06-21
        - Added basin-based composite calculation using longitude-defined basins.
        - Added on-the-fly horizontal-gradient calculation.

"""

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
from tqdm import tqdm
from scipy.stats import ttest_ind, mannwhitneyu # or other tests

# input and output directory
# must check these paths when reproducing composite analysis
composite_dir = '/data/keeling/a/mingshi3/c/xshield/composites'
composite_derived_dir = '/data/keeling/a/mingshi3/c/xshield/Midlatitude-Cyclone-in-X-SHiELD/data/composites'
track_dir = '/data/keeling/a/mingshi3/c/xshield/Midlatitude-Cyclone-in-X-SHiELD/data/tracks'

experiments = [ 'PIRE',
                'PIRE_CO2_1270ppmv',
                'PIRE_PLUS_4K',
                'PIRE_PLUS_4K_CO2_1270ppmv']

# (a) for derived variables, can set levels to [''] and use custom variable name
var_name = 'stability-thetae-200925'
#var_name = 'PRATEsfc'
var_name = 'uv925'
levels = ['']

calc_gradient=False

# (b) or regular definition: variable names and levels, 
#var_name = 'rh'
#levels = [1000,925,850,700,500,200,50]

# Basin. '', 'NP', or 'NA': all, north pacific, north atlantic, it is just longitude subset
basin = ''  

# import and define statistical test function
test = ttest_ind

tk = []
idx = []
for exp in experiments: # load track and indices
    tk.append(np.loadtxt(f'{track_dir}/TK2Y-{exp}/All_Tracks.txt'))
    idx.append(np.loadtxt(f'{track_dir}/TK2Y-{exp}/All_Indexes.txt'))

Y, X = 201,201 # horizontal storm centered grid

  

def get_lon_slice(basin):
    if basin == '':
        return None
    elif basin == 'NP':
        return (120, 240)
    elif basin == 'NA':
        return (280, 360)
    else:
        raise ValueError(f'Unknown basin: {basin}')

def select_tracks(d, basin=''):
    lon = d[:, 2] % 360
    lon_slice = get_lon_slice(basin)

    sel = (
        (d[:, 0] > 2020000000) &
        (d[:, 0] < 2022000000) &
        (d[:, 1] < 65) &
        (d[:, 1] >= 30)
    )

    if lon_slice is not None:
        lon0, lon1 = lon_slice
        sel = sel & (lon >= lon0) & (lon < lon1)

    return np.array(sel)


for level in levels:

    mean_data = [] # 4 exps, Y, X
    p_data = [] # 3 exps (all except CTRL), Y, X

    for i,exp in enumerate(experiments):
        print(level, exp)

        if i == 0:
            # CTRL experiment data: only create composite mean
            d = tk[i]
            sel0 = select_tracks(d, basin=basin)

            data_ctrl = np.load(f'{composite_dir}/TK2Y-{exp}_{var_name}{level}.npy')[sel0].astype(np.float32)
            print(data_ctrl.shape)
            if calc_gradient:
                data_ctrl = np.sqrt(np.gradient(data_ctrl,axis=-1)**2 + np.gradient(data_ctrl,axis=-2)**2) / 25000
            mean_data.append(np.nanmean(data_ctrl,axis=0))
        else:
            # sensitivity experiments: get composite mean and conduct statistical test with CTRL and get p value
            d = tk[i]
            sel = select_tracks(d, basin=basin)

            data_test = np.load(f'{composite_dir}/TK2Y-{exp}_{var_name}{level}.npy')[sel].astype(np.float32)
            if calc_gradient:
                data_test = np.sqrt(np.gradient(data_test,axis=-1)**2 + np.gradient(data_test,axis=-2)**2) / 25000
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
    if calc_gradient:
        np.save(f'{composite_derived_dir}/TK2Y-allexp_means_{basin}{var_name}G{level}_t.npy',np.array(mean_data))
        np.save(f'{composite_derived_dir}/TK2Y-allexp-ctrl_pvalue_{basin}{var_name}G{level}_t.npy',np.array(p_data))
    else:
        np.save(f'{composite_derived_dir}/TK2Y-allexp_means_{basin}{var_name}{level}_t.npy',np.array(mean_data))
        np.save(f'{composite_derived_dir}/TK2Y-allexp-ctrl_pvalue_{basin}{var_name}{level}_t.npy',np.array(p_data))
