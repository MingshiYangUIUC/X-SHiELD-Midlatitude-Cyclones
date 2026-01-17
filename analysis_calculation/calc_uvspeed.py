"""
Author:
Mingshi Yang (mingshi3@illinois.edu)

Date:
2026-01-16

Project:
Characteristics of Midlatitude Cyclones under Climate Change in a Global Storm-Resolving Model

Description:
This script computes wind speed from storm-centered horizontal wind components.
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


level = '925' # hPa

for exp in experiments:

    print(exp)
    u = np.load(f'{composite_dir}/TK2Y-{exp}_u{level}.npy')
    v = np.load(f'{composite_dir}/TK2Y-{exp}_v{level}.npy')

    np.save(f'{composite_dir}/TK2Y-{exp}_uv{level}.npy', np.sqrt(u**2+v**2))
    del u,v

