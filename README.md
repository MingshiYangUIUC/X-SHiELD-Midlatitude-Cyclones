# Characteristics of Extratropical Cyclones under Uniform SST Warming and Increased CO₂ Forcing in a Global Storm-Resolving Model

This repository contains analysis and visualization scripts supporting the manuscript **“Characteristics of Extratropical Cyclones under Uniform SST Warming and Increased CO₂ Forcing in a Global Storm-Resolving Model”** by Mingshi Yang, Zhuo Wang, Lucas Harris, and Kai-Yuan Cheng.

The codebase is organized to separate data retrieval, scientific analysis, and figure generation.

## Repository Structure
- **`analysis_calculation/`**  
  Core analysis scripts, including storm-centered composite construction, derivation and manipulation of additional variables, and statistical significance testing.
  Cyclone track information is derived using the Crawford et al. (2021) cyclone tracking algorithm:  
  https://github.com/alexcrawford0927/cyclonetracking

- **`data/`**  
  Intermediate datasets required to reproduce all figures and results presented in the manuscript. Meteorological fields used to generate cyclone tracks, and generated composite fields are not provided. The provided analysis scripts can be used to regenerate these datasets from archived reanalysis and model outputs following the documented workflow.  
  To reproduce the datasets and figures, users will need to install the required software dependencies and adjust file paths in the scripts to point to the appropriate data storage locations within this repository or to external data sources as needed.

- **`download_preprocess/`**  
  Scripts to retrieve meteorological fields from remote storage and perform basic preprocessing for subsequent analysis.  
  X-SHiELD data are obtained from the GFDL Google Cloud storage archive:  
  https://console.cloud.google.com/storage/browser/gfdl-xshield-pire-2022/X-SHiELD-2021   
  ERA5 Reanalysis data used can be obtained from the Climate Data Store:   
  https://cds.climate.copernicus.eu    

- **`plotting/`**  
  Jupyter notebooks used to generate all the figures and supplementary figures presented
  in the manuscript.

## Software Dependencies

The analysis primarily relies on the following Python packages:
- Python (v3.9.5)
- MetPy (v1.6.1)
- NumPy (v1.26.4)
- Pandas (v2.2.2)
- SciPy (v1.13.1)
- Seaborn (v0.13.2)
- xarray (v2024.7.0)

## Notes

This repository is intended for data and software availability in support of open and reproducible research. Detailed scientific background, methodological discussion, and figure interpretation are provided in the manuscript.

Computing support was provided by NCAR Computational and Information Systems Laboratory (CISL):  
https://www.cisl.ucar.edu/
