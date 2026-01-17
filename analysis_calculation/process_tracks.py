"""
Author:
Mingshi Yang (mingshi3@illinois.edu)

Date:
2026-01-16

Project:
Characteristics of Midlatitude Cyclones under Climate Change in a Global Storm-Resolving Model

Description:
This script reads CSV output from the Crawford et al. (2021) cyclone tracker
(version 13.2, https://github.com/alexcrawford0927/cyclonetracking), and
merge individual track files into a single unified track dataset and
associated index file for subsequent analysis.
"""

import os
import pandas as pd
import numpy as np
import subprocess
from matplotlib import pyplot as plt
import xarray as xr

experiments = [ 'PIRE',
                'PIRE_CO2_1270ppmv',
                'PIRE_PLUS_4K',
                'PIRE_PLUS_4K_CO2_1270ppmv']
import sys
exp = experiments[int(sys.argv[1])]

pd.set_option('display.max_columns', None)

# this is location of tracker output
wd = f'/glade/work/mingshiy/XSHIELD/data/ACTracking-2Y_{exp}/tracker/tracking13_2PTest'

# this is temporary location for individual track processing
wd_tmp = '/glade/work/mingshiy/XSHIELD/tracks/tmp'
subprocess.run(f'rm -rf {wd_tmp}/*',shell=True,check=False)

# location of slp files
wd_slp = f'/glade/work/mingshiy/XSHIELD/data/SLP_{exp}'

print(os.listdir(os.path.join(wd,'CSVSystem')))
NT = 0
for year in sorted(os.listdir(os.path.join(wd,'CSVSystem'))):
    for month in sorted(os.listdir(os.path.join(wd,'CSVSystem',year))):
        p = os.path.join(wd,'CSVSystem',year,month)

        for f in sorted(os.listdir(p)):
            print(NT,f)

            csv = pd.read_csv(os.path.join(p,f))
            data = csv.loc[:,['year','month','day','hour','p_cent','lat','lon']].values
            header = np.int64(data[0][:4])

            file = open(
                f'{wd_tmp}/{exp}_{header[0]}{str(header[1]).zfill(2)}{str(header[2]).zfill(2)}{str(header[3]).zfill(2)}_{NT}.txt',
                'w')
            file.write('PIRE\n')

            for entry in data:
                # sometimes at lysis location there is no SLP data recorded, obtain the SLP from monthly field.
                if np.isnan(entry[4]):
                    print('Found nan at lysis timestep, filling using original data...')
                    slpdata = xr.open_dataset(f'{wd_slp}/_MONTHLY-{header[0]}{str(header[1]).zfill(2)}-psl_1440_720.nc')
                    slpdata['time'] = pd.to_datetime([date.isoformat() for date in slpdata.time.values])
                    fillvalue = slpdata['psl'].sel(
                        time=f'{int(entry[0])}-{str(int(entry[1])).zfill(2)}-{str(int(entry[2])).zfill(2)}T{str(int(entry[3])).zfill(2)}00',
                        latitude=entry[5],longitude=entry[6]%360,
                        method='nearest')
                    slp = float(fillvalue.values) / 100
                    if slp < 500:
                        slp *= 100
                else:
                    slp = entry[4]
                file.writelines(f'{int(entry[0])}{str(int(entry[1])).zfill(2)}{str(int(entry[2])).zfill(2)}{str(int(entry[3])).zfill(2)}'
                                +f'  {round(entry[5],5)}'
                                +f'  {round(entry[6]%360,5)}'
                                +f'  {round(slp,5)}'
                                +'\n'
                )

            file.close()
            NT += 1

print('Number of tracks processed', NT)        
print('Joining Tracks...')

# read list of files (individual tracks)
fs = os.listdir(f'{wd_tmp}')
fs.sort()

# folder to store grouped files
outpath = f'/glade/work/mingshiy/XSHIELD/tracks/TK2Y-{exp}'
if not os.path.isdir(outpath):
    os.makedirs(outpath)

subprocess.run(f'rm -r {outpath}/*',shell=True,check=False)

largefile = open(os.path.join(outpath,'All_Tracks.txt'),'w')
largeidx = open(os.path.join(outpath,'All_Indexes.txt'),'w')

ii = 0
for f in fs:
    lines = open(os.path.join(f'{wd_tmp}',f),'r').readlines()[1:] # first line is a placeholder
    largefile.writelines(lines)
    largeidx.write(f'{ii} {ii+len(lines)}\n')
    ii += len(lines)

largefile.close()
largeidx.close()