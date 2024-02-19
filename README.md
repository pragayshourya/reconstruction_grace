# GRACE-TWS Reconstruction with Deep Learning (Work Under Progress)

## Description:
Welcome to the GRACE-TWS Reconstruction with Deep Learning repository, a dedicated hub for leveraging the power of deep learning models to reconstruct Terrestrial Water Storage (TWS) data from remote sensing datasets. Our repository aims to provide researchers, scientists, and enthusiasts in the field of hydrology and Earth observation with cutting-edge tools and methodologies for accurately estimating TWS using state-of-the-art deep learning techniques.

## Key Objectives:
The primary objective of this repository is to harness the capabilities of deep learning to reconstruct GRACE (Gravity Recovery and Climate Experiment) Terrestrial Water Storage data. GRACE satellites provide critical insights into Earth's water distribution, but their raw data can be noisy and challenging to interpret. By utilising advanced deep learning models, we aim to enhance the accuracy and resolution of TWS estimates derived from GRACE observations.

## Getting Started:
There are two methods to utilize this library. The first approach involves cloning this repository directly, while the second method entails pulling the Docker image. Opting for the Docker option is recommended due to its user-friendly nature. A Docker file will soon be included and maintained within this repository, streamlining the process further.

Different steps to follow for data preparation and model training:

1. **Clipping and Reprojection of Input Datasets**: The initial step involves manipulating the input datasets to fit the desired resolution. This can include tasks such as cropping and reprojecting the input dataset to match the desired resolution. This step is crucial to ensure consistency and compatibility among different input and output datasets.

2. **Conversion to Gridwise Input Files**: After the reprojection process, the modified datasets need to be converted into a format that suits further processing. This often involves organizing the data into a grid-wise dataset, which can be directly used to develop a deep learning model.

3. **Updating 'look_back' Parameter and HPC Submission**: Within the 'main.py' script, there is a parameter called 'look_back.' This parameter likely influences the length of historical data used for making predictions or decisions. Before submitting the script to a High-Performance Computing (HPC) cluster, this 'look_back' parameter must be appropriately adjusted to tailor the analysis to the specific requirements. HPC clusters are powerful computing environments capable of handling complex tasks through parallel processing, which can significantly speed up computations.

4. **Random Search for Hyperparameters**: Hyperparameters are values that influence the behaviour and performance of Machine learning/Deep learning models. The script automates the process of hyperparameter tuning by employing the Random Search method. This technique involves systematically trying out various combinations of hyperparameter values to find those that yield the best model performance.

5. **Saving Results and Predicted Time Series**: The outcomes of training and testing the model are crucial for analysis and evaluation. To keep these results organized, the code creates dedicated directories to store these outputs. Additionally, the predicted time series, which are likely the model's projections or forecasts, are saved in a designated folder named "Time Series." This ensures easy access and reference to these outputs.

6. **Reconstruction Notebook for netCDF Compilation**: Upon the successful execution of 'main.py,' the subsequent 'Reconstruction_notebook.ipynb' contains code to compile a netCDF file. A netCDF (Network Common Data Form) file is a standard format for storing large scientific datasets. The code in the notebook likely not only compiles the file but also adds important attributes to the dataset. These attributes could include metadata, descriptions, units, and other relevant information that provide context to the data and make it more understandable and usable.

## Contributions:
We welcome contributions from any interested parties and are in the process of outlining how to incorporate outside contributions best. For the time being, if you would like to contribute to the development of the library, please contact Pragay Shourya Moudgil (pragay.shourya@gmail.com).

## Credits:
If using the repository for scientific applications, please cite the following: 
1. Moudgil, P. S. and Rao, G. S.: Filling Temporal Gaps within and between GRACE and GRACE-FO Terrestrial Water Storage Changes over Indian Sub-Continent using Deep Learning., EGU General Assembly 2023, Vienna, Austria, 24–28 Apr 2023, EGU23-8218, https://doi.org/10.5194/egusphere-egu23-8218, 2023.
2. Moudgil, P. S., Rao, G.S, Heki, K. (2023). Bridging the Temporal Gaps in GRACE/GRACE-FO Terrestrial Water Storage Anomalies over the Major Indian River Basins using Deep Learning (Under Review).

## Datasets and output:
The processed input datasets can be assessed through the following link: https://www.dropbox.com/scl/fo/nrhbtaqfdn6utkk8psid2/h?rlkey=g0bpowowndyk760v7xj2ehyq8&dl=0

The links to download the original datasets (Please note that datasets are updated regularly. Dates when datasets were assessed are provided in the bracket):
1. CSR-GRACE-TWS (2022-09-05): https://www2.csr.utexas.edu/grace/RL06_mascons.html
2. JPL-GRACE-TWS (2022-09-27): https://grace.jpl.nasa.gov/data/get-data/jpl_global_mascons/
3. GSFZ-GRACE-TWS (2023-01-02): https://earth.gsfc.nasa.gov/geo/data/grace-mascons
4. CHRIPS Monthly Rainfall V2.0 (2022-09-05): https://data.chc.ucsb.edu/products/CHIRPS-2.0/
5. GLDAS-NOAH (2023-01-02): https://disc.gsfc.nasa.gov/datasets/GLDAS_NOAH025_M_2.1/summary
6. ERA5-Temperature (2023-01-02): https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=form
7. MODIS-NDVI (2022-12-31): https://modis.gsfc.nasa.gov/data/dataprod/mod13.php
8. GLDAS-CLSM-(From 2000-01-01 to 2003-02-01) (2023-11-26): https://disc.gsfc.nasa.gov/datasets/GLDAS_CLSM025_D_2.0/summary
9. GLDAS-CLSM-(From 2003-02-01 to 2022-07-01) (2023-11-26): https://disc.gsfc.nasa.gov/datasets/GLDAS_CLSM025_DA1_D_2.2/summary

# References:
1. F Landerer. 2021. TELLUS_GRAC_L3_CSR_RL06_LND_v04. Ver. RL06 v04. PO.DAAC, CA, USA. Dataset accessed [2022-09-08] at https://doi.org/10.5067/TELND-3AC64.
2. Landerer F.W. and S. C. Swenson, Accuracy of scaled GRACE terrestrial water storage estimates. Water Resources Research, Vol 48, W04531, 11 PP, doi:10.1029/2011WR011453, 2012.
3. Loomis et al., 2020, Geophys. Res. Lett., https://doi.org/10.1029/2019GL085488.
4. Sun, Y., R. Riva, and P. Ditmar (2016), Optimizing estimates of annual variations and trends in geocenter motion and J2 from a combination of GRACE data and geophysical models, J. Geophys. Res. Solid Earth, 121, doi:10.1002/2016JB013073.
5. Wahr, J., M. Molenaar, and F. Bryan, Time-variability of the Earth's gravity field: Hydrological and oceanic effects and their possible detection using GRACE, J. Geophys. Res., 103, 32,20530,229, doi:10.1029/98JB02844, 1998.
6. Save, H., S. Bettadpur, and B.D. Tapley (2016), High-resolution CSR GRACE RL05 mascons, J. Geophys. Res. Solid Earth, 121, doi:10.1002/2016JB013007.
7. Save, Himanshu, 2020, "CSR GRACE and GRACE-FO RL06 Mascon Solutions v02", doi: 10.15781/cgq9-nh24.
8. Loomis, B.D., Luthcke, S.B. & Sabaka, T.J. (2019) Regularization and error characterization of GRACE mascons. J Geod 93, 1381–1398. https://doi.org/10.1007/s00190-019-01252-y.
9. Funk, Chris, Pete Peterson, Martin Landsfeld, Diego Pedreros, James Verdin, Shraddhanand Shukla, Gregory Husak, James Rowland, Laura Harrison, Andrew Hoell & Joel Michaelsen. "The climate hazards infrared precipitation with stations-a new environmental record for monitoring extremes". Scientific Data 2, 150066. doi:10.1038/sdata.2015.66 2015.
10. Rodell, M., P.R. Houser, U. Jambor, J. Gottschalck, K. Mitchell, C.-J. Meng, K. Arsenault, B. Cosgrove, J. Radakovich, M. Bosilovich, J.K. Entin, J.P. Walker, D. Lohmann, and D. Toll, The Global Land Data Assimilation System, Bull. Amer. Meteor. Soc., 85(3), 381-394, 2004.
11. Hersbach, H., Bell, B., Berrisford, P., Biavati, G., Horányi, A., Muñoz Sabater, J., Nicolas, J., Peubey, C., Radu, R., Rozum, I., Schepers, D., Simmons, A., Soci, C., Dee, D., Thépaut, J-N. (2023): ERA5 hourly data on single levels from 1940 to the present. Copernicus Climate Change Service (C3S) Climate Data Store (CDS), DOI: 10.24381/cds.adbb2d47 (Accessed on 02-01-2023).
12. Didan, K. (2015). MOD13C2 MODIS/Terra Vegetation Indices Monthly L3 Global 0.05Deg CMG V006 [Data set]. NASA EOSDIS Land Processes Distributed Active Archive Center. Accessed 2022-12-30 from https://doi.org/10.5067/MODIS/MOD13C2.006


