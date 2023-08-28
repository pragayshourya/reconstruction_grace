'''
The Following script is used to convert and save datasets for every grid.
'''

#Imports:
import pandas as pd
import xarray as xr
import numpy as np
import multiprocessing as mp


# Loading datasets from different satellites using netcdf:
grace_jpl = xr.open_dataset('./data/grace_jpl_05.nc')
grace_jpl_df = grace_jpl.to_dataframe().reset_index()
grace_jpl_index = grace_jpl_df[['lat', 'lon']].drop_duplicates().values

# GRACE mascons solutions from different datasets:
grace_jpl = xr.open_dataset('../GRACE/grace_jpl_05.nc')
grace_csr = xr.open_dataset('../GRACE/grace_csr_05.nc').sel(time = slice(None, '2022-08-01'))
grace_gsfz = xr.open_dataset('../GRACE/grace_gsfz_05.nc')
#Other variables import:
gldas_noah = xr.open_dataset('../GLDAS/GLDAS_COMBINED_clipped_india_grace_gridded.nc')                              # GLDAS 
era_temp = xr.open_dataset('../ERA/Temperature_ERA_clipped_india_grace_gridded.nc')                                 # Temperature
modis_ndvi = xr.open_dataset('../MODIS/MODIS_NDVI_05.nc')                                                           # MODIS NDVI
chirps_rain = xr.open_dataset('../CHIRPS/chirps_05.nc')

#Compilation of datasets:
def anomalies(variable):
    return variable-np.mean(np.abs((variable.sel(time=slice("2004-01-01", "2009-12-01")).values)))

def data_prep(i):
    time_section = slice("2001-01-01", "2022-08-01")
    #GRACE:
    lwe_jpl = grace_jpl['lwe_thickness'].sel(lat=i[0], lon=i[1] ,time=time_section).resample(time="1MS").mean() #JPL solutions
    lwe_csr = grace_csr['lwe_thickness'].sel(lat=i[0], lon=i[1] ,time=time_section).resample(time="1MS").mean() #CSR solutions
    lwe_gsfz = grace_gsfz['lwe_thickness'].sel(lat=i[0], lon=i[1] ,time=time_section).resample(time="1MS").mean() # gsfz solutions
    average = (lwe_jpl+lwe_csr+lwe_gsfz)/3
    #Other variables:
    ndvi = anomalies((modis_ndvi['NDVI'].sel(lat=i[0], lon=i[1] ,time=time_section).resample(time="1MS").mean())*0.0001) #NDVI
    temp = anomalies(era_temp['skt'].sel(lat=i[0], lon=i[1] ,time=time_section).resample(time="1MS").mean()) #Skin temperature
    precp = anomalies(chirps_rain['precip'].sel(lat=i[0], lon=i[1] ,time=time_section).resample(time="1MS").mean()) #Rainfall

    #GLDAS variables:
    et = anomalies(gldas_noah['Evap_tavg'].sel(lat=i[0], lon=i[1] ,time=time_section).resample(time="1MS").mean()) #Evapotranspiration
    cws = anomalies(gldas_noah['CanopInt_inst'].sel(lat=i[0], lon=i[1] ,time=time_section).resample(time="1MS").mean()) #Plant canopy 
    runoff = anomalies(gldas_noah['Qs_acc'].sel(lat=i[0], lon=i[1] ,time=time_section).resample(time="1MS").mean()) #Surface runoff
    swe = anomalies(gldas_noah['SWE_inst'].sel(lat=i[0], lon=i[1] ,time=time_section).resample(time="1MS").mean())
    soil_0_10 = anomalies(gldas_noah['SoilMoi0_10cm_inst'].sel(lat=i[0], lon=i[1] ,time=time_section).resample(time="1MS").mean())
    soil_10_40 = anomalies(gldas_noah['SoilMoi10_40cm_inst'].sel(lat=i[0], lon=i[1] ,time=time_section).resample(time="1MS").mean())
    soil_40_100 = anomalies(gldas_noah['SoilMoi40_100cm_inst'].sel(lat=i[0], lon=i[1] ,time=time_section).resample(time="1MS").mean())
    soil_100_200 = anomalies(gldas_noah['SoilMoi100_200cm_inst'].sel(lat=i[0], lon=i[1] ,time=time_section).resample(time="1MS").mean())
    tem_o = pd.DataFrame(list(zip(ndvi.values,
                               temp.values,
                               precp.values,
                               et.values,
                               cws.values,
                               runoff.values,
                               soil_0_10.values,
                               soil_10_40.values,
                               soil_40_100.values,
                               soil_100_200.values,
                               swe.values)), 
                               columns=['NDVI',
                                        'Temperature',
                                        'Precipitation',
                                        'Evapotranspiration',
                                        'Canopy water',
                                        'Runoff',
                                        'SoilM_0_10',
                                        'SoilM_10_40',
                                        'SoilM_40_100',
                                        'SoilM_100_200',
                                        'Snow water equivalent'],
                               index=temp['time'])
    tem_o.index.name = 'Date'
    tem_grace = pd.DataFrame(average.values, columns=['GRACE'], index = lwe_csr['time'])
    tem_grace.index.name = 'Date'
    df = tem_o.merge(tem_grace, how='left', left_index=True, right_index=True)
    df.to_csv('../COMPILED/{}_{}.csv'.format(i[0], i[1]))

if __name__ == '__main__':
    pool = mp.Pool()
    pool.map(data_prep, grace_jpl_index)