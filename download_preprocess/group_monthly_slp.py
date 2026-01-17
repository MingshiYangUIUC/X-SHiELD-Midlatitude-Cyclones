"""
Author:
Mingshi Yang (mingshi3@illinois.edu)

Date:
2026-01-16

Project:
Characteristics of Midlatitude Cyclones under Climate Change in a Global Storm-Resolving Model

Description:
This script aggregates downloaded sea level pressure (SLP) data into monthly
mean files required as input for the Crawford et al. (2021) cyclone tracking
algorithm.
"""

import xarray as xr
from netCDF4 import Dataset
from netCDF4 import num2date
import numpy as np
import pandas as pd
import os
import subprocess

dates = pd.date_range('2019-10-20T00','2022-01-12T00',freq='5d')
date_str = [d.strftime('%Y%m%d%H') for d in dates]

experiments = [ 'PIRE',
                'PIRE_CO2_1270ppmv',
                'PIRE_PLUS_4K',
                'PIRE_PLUS_4K_CO2_1270ppmv']


for exp in experiments[:]:
    if os.path.isdir(f'/glade/work/mingshiy/XSHIELD/data/SLP_{exp}'):
        files = sorted(
            os.path.join(f'/glade/work/mingshiy/XSHIELD/data/SLP_{exp}', f)
            for f in os.listdir(f'/glade/work/mingshiy/XSHIELD/data/SLP_{exp}')
            if '1440x720' in f
        )

        # Open and concat without relying on combine='by_coords'
        datasets = [xr.open_dataset(f) for f in files]
        ds_all = xr.concat(datasets, dim="time")

        # Remove overlapping times across files
        time_index = pd.Index(ds_all.time.values)
        mask = ~time_index.duplicated(keep="first")
        ds_all = ds_all.isel(time=mask)

        print(ds_all.time.values)
        time = pd.to_datetime([date.isoformat() for date in ds_all.time.values])

        for year in time.year.unique():
            for month in time.month.unique():
                # Create a mask for the current year and month
                mask = (time.year == year) & (time.month == month)
                print(year,month,np.sum(mask))
                if np.sum(mask) > 0:
                    ds_month = ds_all.sel(time=mask)
                    ds_month['time'].encoding['dtype'] = 'float64'
                    ds_month.to_netcdf(os.path.join(f'/glade/work/mingshiy/XSHIELD/data/SLP_{exp}',
                        f'_MONTHLY-{year}{str(month).zfill(2)}-psl_1440_720.nc'))
