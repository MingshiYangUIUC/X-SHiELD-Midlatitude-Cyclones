"""
Author:
Mingshi Yang (mingshi3@illinois.edu)

Date:
2026-01-16

Project:
Characteristics of Midlatitude Cyclones under Climate Change in a Global Storm-Resolving Model

Description:
This script uses atmospheric fields with MetPy to compute relative humidity.
"""

from metpy import calc
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
from tqdm import tqdm
from metpy.units import units
import os

composite_dir = '/glade/work/mingshiy/XSHIELD/data/Composites'

experiments = [ 'PIRE',
                'PIRE_CO2_1270ppmv',
                'PIRE_PLUS_4K',
                'PIRE_PLUS_4K_CO2_1270ppmv']

levels = [1000,925,850,700,500,200,50]

for level in levels:
    #level = '850' # hPa

    for exp in experiments:
        print(exp,level)

        q = np.load(f'{composite_dir}/TK2Y-{exp}_q{level}.npy',mmap_mode='r') # kg/kg
        t = np.load(f'{composite_dir}/TK2Y-{exp}_t{level}.npy',mmap_mode='r') # K

        p = np.zeros_like(t) + level

        rh = np.empty_like(q)

        num_chunks = 50
        chunk_size = q.shape[0] // num_chunks

        for i in tqdm(range(num_chunks)):

            # Calculate start and end indices for the current chunk
            start_idx = i * chunk_size
            end_idx = (i + 1) * chunk_size if i < num_chunks - 1 else q.shape[0]
            
            # Process the current chunk
            q_chunk = q[start_idx:end_idx]
            t_chunk = t[start_idx:end_idx]
            p_chunk = p[start_idx:end_idx]
            
            rh_chunk = calc.relative_humidity_from_specific_humidity(p_chunk * units.hPa, t_chunk * units.degK, specific_humidity=q_chunk * units('kg/kg'))
            
            # Store the theta_e results back into the respective array
            rh[start_idx:end_idx] = rh_chunk.magnitude  # Assuming you want to store without units

            del q_chunk, t_chunk, p_chunk, rh_chunk
        # Save the complete theta_e array to disk
        np.save(f'{composite_dir}/TK2Y-{exp}_rh{level}.npy', rh)
        del q, t, rh

