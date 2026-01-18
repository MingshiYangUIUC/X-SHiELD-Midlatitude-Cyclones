"""
Author:
Mingshi Yang (mingshi3@illinois.edu)

Date:
2026-01-16

Project:
Characteristics of Midlatitude Cyclones under Climate Change in a Global Storm-Resolving Model

Description:
This script downloads meteorological data from the X-SHiELD Google
Cloud storage archive (using wget).
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

# because host files may cover different number of days, we try every day by checking
# whether a file exist in that day, and download all existing files
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

# set where to store data (large disk)
outdir = '/glade/work/mingshiy/XSHIELD/data/metfields'
os.makedirs(outdir, exist_ok=True)

# all dates
dates = pd.date_range('2019-10-20T00','2022-01-12T00',freq='1d')
date_str = [d.strftime('%Y%m%d%H') for d in dates]

# all experiments
experiments = [ 'PIRE',
                'PIRE_CO2_1270ppmv',
                'PIRE_PLUS_4K',
                'PIRE_PLUS_4K_CO2_1270ppmv']

# please consult description of data in X-SHiELD storage website
variables = ['omg','t','q', 'z','vort'] # a list of variable names
levels = ['1000','925','850','700','500','200','50'] # a list of pressure levels or "sfc" for surface variables


# download the file by looping through all levels, experiments, dates, and variables
for plevel in levels:
    for exp in experiments[:]:
        if not os.path.isdir(os.path.join(outdir,f'{exp}')):
            os.mkdir(os.path.join(outdir,f'{exp}'))
        for dt in date_str:
            #print(exp,dt)
            for var in variables:
                subprocess.run(f'echo "{exp} {dt} {var} {plevel}"',shell=True,check=False)
                url = f'https://storage.googleapis.com/gfdl-xshield-pire-2022/X-SHiELD-2021/{exp}/{dt}/{var}{plevel}_coarse_C3072_1440x720.fre.nc'

                outpath = f'{os.path.join(outdir,f"{exp}",f"{exp}-{dt}-{var}{plevel}_coarse_C3072_1440x720.fre.nc")}'

                if is_valid_url(url) and not os.path.isfile(outpath):
                    print('  Get File',outpath,flush=True)
                    subprocess.run(f'wget {url}'
                        + f' -O {os.path.join(outdir,f"{exp}",f"{exp}-{dt}-{var}{plevel}_coarse_C3072_1440x720.fre.nc")} -q',shell=True,check=False)
                    
                else:
                    if is_valid_url(url):
                        print('  File Exist',exp,dt,var,flush=True)
                    else:
                        print('  File (date) Invalid',exp,dt,var,flush=True)