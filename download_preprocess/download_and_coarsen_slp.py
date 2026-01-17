"""
Author:
Mingshi Yang (mingshi3@illinois.edu)

Date:
2026-01-16

Project:
Characteristics of Midlatitude Cyclones under Climate Change in a Global Storm-Resolving Model

Description:
This script downloads sea level pressure (SLP) data from the X-SHiELD Google
Cloud storage archive (using wget) and spatially coarsens the data to 0.25° resolution for
consistency with other analysis fields.
"""

import datetime
import pandas as pd
import subprocess
import os


import xarray as xr
from netCDF4 import Dataset
from netCDF4 import num2date
import numpy as np
import pandas as pd
import os
import subprocess
import requests

def is_valid_url(url):
    try:
        response = requests.head(url)
        # Check if the response status code is OK (200) and Content-Type header exists
        if response.status_code == 200 and 'Content-Type' in response.headers:
            return True
        else:
            return False
    except requests.RequestException as e:
        print(f"Error checking URL: {e}")
        return False

# all dates
dates = pd.date_range('2019-10-20T00','2022-01-12T00',freq='1d')
date_str = [d.strftime('%Y%m%d%H') for d in dates]

# all experiments
experiments = [ 'PIRE',
                'PIRE_CO2_1270ppmv',
                'PIRE_PLUS_4K',
                'PIRE_PLUS_4K_CO2_1270ppmv']

# set where to store data (large disk)
outdir = '/glade/work/mingshiy/XSHIELD/data'
os.makedirs(outdir, exist_ok=True)


# download the file by looping through all experiments and dates
for exp in experiments[:]:
    if not os.path.isdir(os.path.join(outdir,f'SLP_{exp}')):
        os.mkdir(os.path.join(outdir,f'SLP_{exp}'))
    for dt in date_str:
        #print(exp,dt)
        subprocess.run(f'echo "{exp} {dt}"',shell=True,check=False)
        url = f'https://storage.googleapis.com/gfdl-xshield-pire-2022/X-SHiELD-2021/{exp}/{dt}/psl_C3072_11520x5760.fre.nc'
        valid = False

        inpath = f'/glade/work/mingshiy/XSHIELD/data/SLP_{exp}/{dt}-psl_C3072_11520x5760.fre.nc'
        outpath = f'/glade/work/mingshiy/XSHIELD/data/SLP_{exp}/{dt}-psl_C3072_1440x720.fre.nc'

        if (not (os.path.isfile(inpath) or os.path.isfile(outpath))) and is_valid_url(url):
            subprocess.run(f'wget {url}'
                + f' -O {os.path.join(outdir,f"SLP_{exp}",f"{dt}-psl_C3072_11520x5760.fre.nc")} -q',shell=True,check=False)
            valid = True
        else:
            subprocess.run(f'echo "File exists or Date invalid"',shell=True,check=False)
        
        if valid: # a valid file either exists or is just retrieved
            
            check = os.path.isfile(outpath)
            subprocess.run(f'echo "{exp} {dt} {check}"',shell=True,check=False)
            #print(check)
            if not check: # if target does not exist, coarsen source file

                ds = Dataset(inpath, 'r')

                psl_var = ds.variables['psl']
                lat = ds.variables['grid_yt']
                lon = ds.variables['grid_xt']
                t = ds.variables['time']

                dates = num2date(t[:], units=t.units, calendar=t.calendar)

                lat_new = (lat[3::8]+lat[4::8])/2
                lon_new = (lon[3::8]+lon[4::8])/2

                psl_a = psl_var[:, 3::8, 3::8]
                psl_b = psl_var[:, 3::8, 4::8]
                psl_c = psl_var[:, 4::8, 3::8]
                psl_d = psl_var[:, 4::8, 4::8]

                psl_mean = np.mean(np.stack((psl_a, psl_b, psl_c, psl_d), axis=0), axis=0)

                ds_out = xr.Dataset(
                    {
                        "psl": (("time", "latitude", "longitude"), np.array(psl_mean))
                    },
                    coords={
                        "time": dates,
                        "latitude": lat_new,
                        "longitude": lon_new
                    }
                )

                ds_out.to_netcdf(outpath)
                ds.close()

                subprocess.run(f'echo "Processed {exp} {dt} {outpath}"',shell=True,check=False)
