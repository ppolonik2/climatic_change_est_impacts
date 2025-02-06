"""
This script calculate the LENS metrics. Due to large file size of the LENS2 dataset
   we do not include it with our data. The file name is therefore a placeholder
   for regidded LENS2 temperature and precipitation data.
The output of this script is included in our data download.
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from scipy.stats import linregress
import multiprocessing
import os

usevar = 'tas' #   tas or pr
if usevar == 'pr':
    lensvar = 'PRECT'
elif usevar == 'tas':
    lensvar = 'TREFHT'

lens = xr.open_dataset('TREFHT_PRECT_05deg.nc')  # This dataset is large and not included (~25 GB)
lens = lens.assign_coords({'time':lens['time.year']})
lens = lens.rename({'time':'year'})
lens = lens.sel({'year':range(1960,2051)})

lens['TREFHT_ensmean'] = lens['TREFHT'].mean('ens')
lens['PRECT_ensmean']  = lens['PRECT'].mean('ens')

if '{}_slopes_lens05.nc'.format(usevar) not in os.listdir():
    def calculate_slope(y):
        x = range(len(y))
        #   slope, _, _, _, _ = linregress(x, y)
        regs = np.polyfit(x,y,1)
        slope = regs[0]
        return slope
    
    def calculate_intercept(y):
        x = range(len(y))
        #   slope, _, _, _, _ = linregress(x, y)
        regs = np.polyfit(x,y,1)
        slope = regs[1]
        return slope
    
    def calculate_shock(y):
        x = range(len(y))
        regs = np.polyfit(x,y,1)
        vals = np.polyval(regs,x)
        resid = y - vals
        return np.var(resid)
    
    def calculate_slope2(y, axis):
        return np.apply_along_axis(calculate_slope, -1, y)
    
    def calculate_shock2(y, axis):
        return np.apply_along_axis(calculate_shock, -1, y)
    
    window_size = 10
    if usevar == 'tas':
        rollingm = lens['TREFHT_ensmean'].rolling(year=window_size, center=False)
    elif usevar == 'pr':
        rollingm = lens['PRECT_ensmean'].rolling(year=window_size, center=False)
    
    slope_data = rollingm.reduce(calculate_slope2)

    #   Shock as dif from ensemble mean, variance, averaged
    if usevar == 'tas':
        anom = lens['TREFHT'] - lens['TREFHT_ensmean']
    elif usevar == 'pr':
        anom = lens['PRECT'] - lens['PRECT_ensmean']

    rolling = anom.rolling(year=window_size, center=False)
    rollvar = rolling.var('year')
    shock_data = rollvar.mean('ens')
    
    slope_data.to_netcdf('{}_slopes_lens05.nc'.format(usevar))
    shock_data.to_netcdf('{}_shocks_lens05.nc'.format(usevar))

else:
    slope_data = xr.open_dataarray('../data/CMIP6/CESM2/{}_slopes_lens05.nc'.format(usevar))
    shock_data = xr.open_dataarray('../data/CMIP6/CESM2/{}_shocks_lens05.nc'.format(usevar))



