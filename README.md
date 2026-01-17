# Midlatitude-Cyclone-in-X-SHiELD

This repository contains analysis and visualization scripts supporting the
manuscript **“Characteristics of Midlatitude Cyclones under Climate Change in a
Global Storm-Resolving Model.”**

The codebase is organized to separate data retrieval, scientific analysis, and
figure generation.

## Repository Structure

- **`download_preprocess/`**  
  Scripts to retrieve meteorological fields from remote storage and
  perform basic preprocessing for subsequent analysis.  
  X-SHiELD data are obtained from the GFDL Google Cloud storage archive:  
  https://console.cloud.google.com/storage/browser/gfdl-xshield-pire-2022/X-SHiELD-2021

- **`analysis_calculation/`**  
  Core analysis scripts, including storm-centered composite construction,
  derivation and manipulation of additional variables, and statistical
  significance testing.  
  Cyclone track information is derived using the Crawford et al. (2021) cyclone
  tracking algorithm:  
  https://github.com/alexcrawford0927/cyclonetracking

- **`plotting/`**  
  Jupyter notebooks used to generate figures and supplementary figures presented
  in the manuscript.

## Software Dependencies

The analysis primarily relies on the following Python packages:
- Python (v3.9.5)
- MetPy (v1.6.1)
- NumPy (v1.26.4)
- SciPy (v1.13.1)
- xarray (v2024.7.0)

## Notes

This repository is intended for data and software availability in support of
open and reproducible research. Detailed scientific background, methodological
discussion, and figure interpretation are provided in the manuscript.

Computing support was provided by NCAR Computational and Information
Systems Laboratory (CISL):  
https://www.cisl.ucar.edu/
