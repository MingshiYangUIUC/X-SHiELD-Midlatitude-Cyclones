"""
Author:
Mingshi Yang (mingshi3@illinois.edu)

Date:
2026-01-16

Project:
Characteristics of Midlatitude Cyclones under Climate Change in a Global Storm-Resolving Model

Description:
This script constructs storm-centered composite datasets by interpolating model lat-
lon fields onto a storm-centric grid using horizontal bilinear interpolation. The
storm-centered compositing framework is adapted from Stoll et al. (2021,
https://doi.org/10.5194/wcd-2-19-2021).
"""

import xarray as xr
import numpy as np
from matplotlib import pyplot as plt
import helper_centered_grid as cg
import os
from datetime import datetime
import pandas as pd

from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import psutil
import gc

# input and output directory
metfield_dir = '/glade/work/mingshiy/XSHIELD/data/metfields'
composite_dir = '/glade/work/mingshiy/XSHIELD/data/Composites'

# these dates are used to identify and find downloaded files
dates_coarse = pd.date_range('2019-10-20T00','2021-01-12T00',freq='5d')
date_str_coarse = [d.strftime('%Y%m%d%H') for d in dates_coarse]

dates = pd.date_range('2019-10-20T00','2021-01-12T00',freq='1d')
date_str = [d.strftime('%Y%m%d%H') for d in dates]

experiments = [ 'PIRE',
                'PIRE_CO2_1270ppmv',
                'PIRE_PLUS_4K',
                'PIRE_PLUS_4K_CO2_1270ppmv']

np.seterr(all='ignore')

def sqr_reshape(data): # reshape so that data is standard N-S/W-E dimension similar to latlon.
    # plot it with coordinate x (0,1,2...) and y(90,89,88..) meshgrid
    s = int(len(data)**0.5)
    data = data.reshape((s,s))[::-1,::-1].T
    return data

# bilinear interpolation as weighted average of four neighbouring points
# this function computes weights given the target grid
def get_weight_idx_from_grid_bi_evengrid025(centered_grid):
    L = len(centered_grid)

    x = centered_grid[:,:,0]
    y = centered_grid[:,:,1]
    x1,x2 = np.floor((x+0.125)*4)/4-0.125,np.ceil((x+0.125)*4)/4-0.125
    y1,y2 = np.floor((y+0.125)*4)/4-0.125,np.ceil((y+0.125)*4)/4-0.125
    ix1,ix2 = np.int16((x1-0.125)/0.25),np.int16((x2-0.125)/0.25)
    iy1,iy2 = np.int16((90-y1-0.125)/0.25),np.int16((90-y2-0.125)/0.25)

    w11 = (x2-x)*(y2-y)/((x2-x1)*(y2-y1))
    w12 = (x2-x)*(y-y1)/((x2-x1)*(y2-y1))
    w21 = (x-x1)*(y2-y)/((x2-x1)*(y2-y1))
    w22 = (x-x1)*(y-y1)/((x2-x1)*(y2-y1))
    
    w11[x1==x2] = ((y2-y)/(y2-y1))[x1==x2]
    w12[x1==x2] = ((y-y1)/(y2-y1))[x1==x2]
    w21[x1==x2] = 0
    w22[x1==x2] = 0
    hl = int((len(centered_grid)-1)/2)
    w11[hl,hl] = 1
    w12[hl,hl] = 0
    #print(np.min(ix1),np.max(ix1))
    return w11,w12,w21,w22,ix1%1440,ix2%1440,iy1,iy2


# need this func because number of dates in each downloaded file may be different.
def find_filedate(exp,var_name,level,d_str): # return date of the correct file
    if os.path.isfile(
        f'{metfield_dir}/{exp}/{exp}-{d_str}-{var_name}{level}_coarse_C3072_1440x720.fre.nc'
        ):
        return d_str
    else:
        for i in range(5):
            newdstr = (datetime.strptime(d_str,'%Y%m%d%H') - pd.DateOffset(days=i)).strftime('%Y%m%d%H')
            if os.path.isfile(
                f'{metfield_dir}/{exp}/{exp}-{newdstr}-{var_name}{level}_coarse_C3072_1440x720.fre.nc'
                ):
                break
        #print(newdstr)
        return newdstr


def worker_func(ilist):
    #print(ilist)
    #quit()
    result_chunk = []
    for i in ilist:

        center_lat = Lats[np.argmin((Lats-TK[i][1])**2)]
        center_lon = Lons[np.argmin((Lons-TK[i][2]%360)**2)]
        
        lons,lats,grid = cg.get_centered_lonlat(center_lon,center_lat,2500,25,True,'new',0)
        latmin,latmax = np.floor(np.min(lats))-0.125,min(np.ceil(np.max(lats))+0.125,89.875)

        timestep = str(np.int64(TK[i][0]))

        var_itp = var.sel(time=f'{timestep[0:4]}-{timestep[4:6]}-{timestep[6:8]}T{timestep[8:10]}',grid_yt_coarse=slice(latmin,latmax)).values[0,::-1] # y top down

        latshift = np.int16((89.875-latmax)/0.25)
        
        w11,w12,w21,w22,ix1,ix2,iy1,iy2 = get_weight_idx_from_grid_bi_evengrid025(grid)

        iy1 -= latshift
        iy2 -= latshift

        center_data_var = cg.get_centered_data_with_weight(var_itp,w11,w12,w21,w22,ix1,ix2,iy1,iy2)

        var_out = sqr_reshape(center_data_var).astype(np.float32)

        mem = psutil.virtual_memory()

        # progress and memory monitor
        percentage = int(round((i-ilist[0])*100/len(ilist)))
        print('-'*percentage+'_'*(100-percentage)+f' {round(mem.available/1024**3,2)}G     ',end='\r')
        result_chunk.append(var_out)
    #print(len(result_chunk))
    return result_chunk


if __name__ == "__main__":
    print(cpu_count())

    for exp in experiments[:]:
        print(exp)
 
        # define levels and variables that need process
        levels = ['1000','925','850','700','500','200','50']
        var_names = ['q','t','omg']

        varlev = [(i,j) for i in var_names for j in levels]
        print(varlev)

        for var_name, level in varlev:

            print(exp, var_name, level)

            if os.path.isfile(os.path.join(composite_dir,f'TK2Y-{exp}_{var_name}{level}.npy')):
                print('Destination file exists.')
                continue
            
            # load the track
            TK = np.loadtxt(f'/glade/work/mingshiy/XSHIELD/tracks/TK2Y-{exp}/All_Tracks.txt')

            Lats = np.linspace(-89.875,89.875,720)
            Lons = np.linspace(0.125,359.875,1440)

            # open all dataset
            var_ds = [os.path.join(f'{metfield_dir}/{exp}',f) for f in os.listdir(f'{metfield_dir}/{exp}') if f'-{var_name}{level}_' in f]
            var_ds.sort()

            datasets = [xr.open_dataset(f) for f in var_ds]
            ds_all = xr.concat(datasets, dim="time")

            # Remove overlapping times across files
            time_index = pd.Index(ds_all.time.values)
            mask = ~time_index.duplicated(keep="first")
            ds_all = ds_all.isel(time=mask)

            var = ds_all[f'{var_name}{level}_coarse']

            data_field = np.zeros((len(TK),201,201),dtype=np.float32)
            print(data_field.nbytes/1024**3)

            # parallel processing with multiprocessing pool
            full_ilist = np.arange(len(TK))
            
            nproc = 32
            chunk = np.int64(np.round(np.linspace(0,len(TK),nproc+1),0))
            with Pool(processes=min(nproc,cpu_count())) as pool:  # Adjust the number of processes as appropriate
                results = pool.map(worker_func, [np.arange(chunk[i],chunk[i+1]) for i in range(nproc)])
            
            data_field = np.concatenate(results, axis=0).astype(np.float32)
            print('Saving')

            np.save(os.path.join(composite_dir,f'TK2Y-{exp}_{var_name}{level}.npy'),data_field)

            del data_field, results, var, TK
            gc.collect()

            print('')