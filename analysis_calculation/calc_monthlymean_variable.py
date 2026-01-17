"""
Author:
Mingshi Yang (mingshi3@illinois.edu)

Date:
2026-01-16

Project:
Characteristics of Midlatitude Cyclones under Climate Change in a Global Storm-Resolving Model

Description:
This script computes monthly mean fields from full time series data, with a
primary application to 200 hPa zonal wind. The processed output is used for
Figure 2 in the manuscript.
"""

import xarray as xr
import numpy as np
from matplotlib import pyplot as plt
import sys
import os
from datetime import datetime
import pandas as pd

from tqdm import tqdm
from scipy.stats import ttest_ind

metfield_dir = '/glade/work/mingshiy/XSHIELD/data/metfields'

dates_coarse = pd.date_range('2019-10-20T00','2021-01-12T00',freq='5d')
date_str_coarse = [d.strftime('%Y%m%d%H') for d in dates_coarse]

dates = pd.date_range('2019-10-20T00','2021-01-12T00',freq='1d')
date_str = [d.strftime('%Y%m%d%H') for d in dates]

experiments = [ 'PIRE',
                'PIRE_CO2_1270ppmv',
                'PIRE_PLUS_4K',
                'PIRE_PLUS_4K_CO2_1270ppmv']

# process U200 or other variables
var_names = ['u',]
levels = ['200',]

ctrl = []

for exp in experiments:
    for var_name in var_names:
        for level in levels:
            if not os.path.isdir(f'{metfield_dir}/{exp}_monthlymean'):
                os.mkdir(f'{metfield_dir}/{exp}_monthlymean')
            print(exp,var_name,level)
            files = os.listdir(f'{metfield_dir}/{exp}')
            files = sorted([os.path.join(f'{metfield_dir}/{exp}',f) for f in files if f'-{var_name}{level}' in f])
            files.sort()
            print(files)
            #quit()
            #ds_all = xr.open_mfdataset(files)
            # Open and concat without relying on combine='by_coords'
            datasets = [xr.open_dataset(f) for f in files]
            ds_all = xr.concat(datasets, dim="time")

            # Remove overlapping times across files
            time_index = pd.Index(ds_all.time.values)
            mask = ~time_index.duplicated(keep="first")
            ds_all = ds_all.isel(time=mask)
            #print(ds_all.time.values)
            #quit()
            time = pd.to_datetime([date.isoformat() for date in ds_all.time.values])

            i_mon = 0

            for year in time.year.unique():
                for month in time.month.unique():
                    # Create a mask for the current year and month
                    mask = (time.year == year) & (time.month == month)
                    print(year,month,np.sum(mask))

                    if np.sum(mask) > 0:
                        if exp == 'PIRE':
                            print('Keeping full field')
                            ctrl.append(ds_all.sel(time=mask)['u200_coarse'].values)
                            #print(ctrl[-1].shape)
                        else:
                            # compare with ctrl: t test
                            print('Calculating significance')
                            csim = ctrl[i_mon]
                            sim = ds_all.sel(time=mask)['u200_coarse'].values
                            Y = csim.shape[-2]
                            X = csim.shape[-1]

                            p_values = np.zeros((Y, X))
                            for i in tqdm(range(Y//2)): # don't process Southern Hemisphere
                                for j in range(X):
                                    t_stat, p_value = ttest_ind(csim[:, i, j], sim[:, i, j],nan_policy='omit')
                                    #t_stats[i, j] = t_stat
                                    p_values[i, j] = p_value
                                    if np.isnan(p_value):
                                        print('nan!',i,j)
                            
                            np.save(os.path.join(f'{metfield_dir}/{exp}_monthlymean',
                                    f'{year}{str(month).zfill(2)}-pvalue-{var_name}{level}_coarse.npy'),p_values)
                            del csim, sim

                        print('Process monthly mean field')
                        ds_month = ds_all.sel(time=mask).mean(dim='time')
                        #ds_month['time'].encoding['dtype'] = 'float64'
                        ds_month.to_netcdf(os.path.join(f'{metfield_dir}/{exp}_monthlymean',
                            f'{year}{str(month).zfill(2)}-{var_name}{level}_coarse.nc'))

                        del mask, ds_month

                        i_mon += 1
