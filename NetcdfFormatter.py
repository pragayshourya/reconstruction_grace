'''
This module is indented to reproject the netcdf datasets available into desired resolution.
'''

#Imports
import pandas as pd
import xarray as xr
import numpy as np
import cdo
import regionmask
import geopandas as gpd
import os
import glob



class NetcdfFormatter:
    def __init__(self):
        self.cdo_lib = cdo.Cdo()

    def clip_region(self, path='', shp_f='', save_loc = ''):
        if os.path.exists('{}{}_clipped.nc'.format(save_loc, path.split('/')[-1][:-2])):
            return
        if type(path) == str:
            ds = xr.open_dataset(path)
        elif type(path) == list:
            ds = xr.open_mfdataset(path)
        else:
            raise ValueError("Please enter a valid path. Path can be a string or list of paths but not {}.".format(type(path)))
        
        shp = gpd.read_file(shp_f)
        min_lon, min_lat, max_lon, max_lat = shp.total_bounds
        
        lat_name, lon_name = None, None


        for var_name in ds.variables:
            if 'Units' in ds[var_name].attrs or 'units' in ds[var_name].attrs:
                try:
                    units = ds[var_name].attrs['Units']
                except:
                    units = ds[var_name].attrs['units']
                if 'lat' in var_name and 'bound' not in var_name.lower() and units.lower() in ['degrees_north', 'degree_north', 'degrees_n', 'degree_n', 'degreesn', 'degreen']:
                    lat_name = var_name
                if 'lon' in var_name and 'bound' not in var_name.lower() and units.lower() in ['degrees_east', 'degree_east', 'degrees_e', 'degree_e', 'degreese', 'degreee']:
                    lon_name = var_name
                if lat_name is not None and lon_name is not None:
                    break
        
        if lat_name is None or lon_name is None:
            raise ValueError("Could not automatically detect latitude and longitude variable names.")
        
        ds = ds.sel(**{lat_name: slice(min_lat, max_lat), lon_name: slice(min_lon, max_lon)})
        poly = regionmask.Regions(np.array(shp.geometry))
        mask = poly.mask(ds.isel(time=0), lat_name=lat_name, lon_name=lon_name)
        ds = ds.where(mask == 0)
        ds.to_netcdf('{}{}_clipped.nc'.format(save_loc, path.split('/')[-1][:-2]))
        # ds.close()
        return None



        
    def reproject_nc(self, algorithm = 'bil', remap_grid = '', path = '', save_loc = ''):
        output_f = '{}{}_rep.nc'.format(save_loc, path.split('/')[-1][:-2])
        # if os.path.exists(output_f):
            # return
        options = "remap{},{} {} {}".format(algorithm, remap_grid, path, output_f)
        # print(options)
        self.cdo_lib.run(options)
        self.cdo_lib.cleanTempDir()
        return None


