"""
Author:
Mingshi Yang (mingshi3@illinois.edu)

Date:
2026-01-16

Project:
Characteristics of Midlatitude Cyclones under Climate Change in a Global Storm-Resolving Model

Description:
This script computes equivalent potential temperature (θₑ) for all experiments
using MetPy and derives a measure of static stability based on the vertical
difference in θₑ between 200 hPa and 925 hPa.
"""

from metpy import calc
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
from tqdm import tqdm
from metpy.units import units
import os

# input and output dirs will be the same
composite_dir = '/glade/work/mingshiy/XSHIELD/data/Composites'

experiments = [ 'PIRE',
                'PIRE_CO2_1270ppmv',
                'PIRE_PLUS_4K',
                'PIRE_PLUS_4K_CO2_1270ppmv']

# calc static stability (difference between 200 hPa and 925 hPa)
# run this after both 925 hPa and 200 hPa levels are processed
calc_static_stability = True
if calc_static_stability:
    for exp in experiments:
        print(exp)
        sta = np.load(f'{composite_dir}/TK2Y-{exp}_thetae200.npy') \
            - np.load(f'{composite_dir}/TK2Y-{exp}_thetae925.npy')
        np.save(f'{composite_dir}/TK2Y-{exp}_stability-thetae-200925.npy', sta)

    quit() # no recalculation of theta_e if static stability can be derived successfully
else:
    pass

# calculate theta_e using metpy
level = '925' # hPa

for exp in experiments:
    print(exp)
    q = np.load(f'{composite_dir}/TK2Y-{exp}_q{level}.npy',mmap_mode='r').astype(np.float32) # kg/kg
    t = np.load(f'{composite_dir}/TK2Y-{exp}_t{level}.npy',mmap_mode='r').astype(np.float32) # K

    theta_e = np.empty_like(q)

    num_chunks = 50
    chunk_size = q.shape[0] // num_chunks

    for i in tqdm(range(num_chunks)):

        # Calculate start and end indices for the current chunk
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size if i < num_chunks - 1 else q.shape[0]
        
        # Process the current chunk
        q_chunk = q[start_idx:end_idx]
        t_chunk = t[start_idx:end_idx]
        
        # Calculate dewpoint temperature from specific humidity for the chunk
        td_chunk = calc.dewpoint_from_specific_humidity(float(level) * units.hPa, specific_humidity=q_chunk * units('kg/kg'))
        
        # Calculate equivalent potential temperature for the chunk
        theta_e_chunk = calc.equivalent_potential_temperature(float(level) * units.hPa, t_chunk * units.degK, td_chunk)
        
        # Store the theta_e results back into the respective array
        theta_e[start_idx:end_idx] = theta_e_chunk.magnitude  # Assuming you want to store without units

        del q_chunk, t_chunk, td_chunk, theta_e_chunk
    # Save the complete theta_e array to disk
    np.save(f'{composite_dir}/TK2Y-{exp}_thetae{level}.npy', theta_e)
    del q, t, theta_e

