"""
Author:
    Mingshi Yang (mingshi3@illinois.edu)

Date:
    2026-06-21

Project:
    Characteristics of Midlatitude Cyclones under Climate Change in a Global Storm-Resolving Model

Description:
    Created during Revision 1 to merge segmented X-SHiELD specific humidity
    NetCDF files by experiment and pressure level. The script identifies available
    q files, reports their total size, and uses CDO mergetime to produce merged
    time-series NetCDF files. Serves to improve I/O efficiency of subsequent analyses.
"""

import os
import glob
import subprocess

# input and output directory
data_dir = '/data/keeling/a/mingshi3/c/xshield/metfields'

experiments = [
    'PIRE',
    'PIRE_CO2_1270ppmv',
    'PIRE_PLUS_4K',
    'PIRE_PLUS_4K_CO2_1270ppmv'
]

levels = ['1000', '925', '850', '700', '500', '200', '50']

for exp in experiments:
    for level in levels:

        all_files = sorted(
            glob.glob(
                os.path.join(
                    data_dir,
                    exp,
                    f'{exp}-*-q{level}_*.nc'
                )
            )
        )
        print(exp, level, len(all_files))

        if len(all_files) == 0:
            print(f'No files found for {exp} q{level}')
            continue

        output_file = os.path.join(
            data_dir,
            exp,
            f'{exp}-merged-q{level}.nc'
        )

        if os.path.isfile(output_file):
            print(f'Output {output_file} exists.')
            continue

        total_size_bytes = sum(os.path.getsize(f) for f in all_files)

        total_size_gb = total_size_bytes / 1024**3
        total_size_tb = total_size_bytes / 1024**4

        print(
            exp,
            level,
            len(all_files),
            f'{total_size_gb:.2f} GB',
            f'({total_size_tb:.2f} TB)'
        )

        cmd = [
            'cdo',
            '-O',
            'mergetime',
            *all_files,
            output_file
        ]

        print('Running:', ' '.join(cmd))
        subprocess.run(cmd, check=True)
