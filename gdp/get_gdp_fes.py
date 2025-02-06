import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import xarray as xr
import subprocess
import statsmodels.formula.api as smf
import seaborn as sns
import geopandas as gpd
import cartopy.crs as ccrs
import time
import pdb

#  Options, but no guarantee that every combination works
#  Script shows how we execute some formatting and regressions
aggunits = ['pixel','admin_id','country_id']
growth = True
anoms  = False

combinedxr = xr.open_dataset('../data/gdp/combxr_05deg2.nc')

combinedxr = combinedxr.where(combinedxr['pop']>0)
combinedpd = combinedxr.to_dataframe()
combinedpd = combinedpd.dropna()
combinedpd_orig = combinedpd.copy()

def get_coefs(formula,data,return_errors=False):
    mod = smf.ols(formula,data)
    res = mod.fit()

    if not return_errors:
        return res.params
    else:
        return res.params, res.bse

#   Turn xarray into pandas and weight T and P by population to aggregate
combinedpds = {}
for aggunit in aggunits:
    combinedpd = combinedpd_orig.copy()
    if aggunit != 'pixel':
        weightvar = aggunit
        if not aggunit.endswith('_id'):
            aggunit += '_id'
        weightvar = aggunit
    
        combinedpd['weightedT'] = combinedpd['T']*combinedpd['pop']
        combinedpd['weightedP'] = combinedpd['P']*combinedpd['pop']
        combinedpd['weightedT50'] = combinedpd['T_2050']*combinedpd['pop']
        combinedpd['weightedP50'] = combinedpd['P_2050']*combinedpd['pop']
        combinedpd['weightedgdp'] = combinedpd['gdppc']*combinedpd['pop']
        dfgrp = combinedpd.groupby([weightvar,'time']).sum()
        Tw = dfgrp['weightedT']/dfgrp['pop']
        Pw = dfgrp['weightedP']/dfgrp['pop']
        Tw50 = dfgrp['weightedT50']/dfgrp['pop']
        Pw50 = dfgrp['weightedP50']/dfgrp['pop']
        gdpw = dfgrp['weightedgdp']/dfgrp['pop']
    
        combinedpdw = pd.DataFrame([gdpw,Tw,Pw,Tw50,Pw50],
                                        index=['gdppc','T','P','T_2050','P_2050']).T

        combinedpdw['T2'] = combinedpdw['T']**2
        combinedpdw['P2'] = combinedpdw['P']**2
        combinedpdw['T2_2050'] = combinedpdw['T_2050']**2
        combinedpdw['P2_2050'] = combinedpdw['P_2050']**2
        if 'Tabs' in combinedpd.columns:
            combinedpdw['Tabs2'] = combinedpdw['Tabs']**2
            combinedpdw['Pabs2'] = combinedpdw['Pabs']**2

        combinedpdw['pop'] = dfgrp['pop']
        combinedpdw = combinedpdw.reset_index()
    
        if growth:
            gdptab  = np.log(combinedpdw.pivot(index='time',columns=aggunit,values='gdppc'))
            gdptabg = gdptab.diff(1).stack()
            gdptabg.name = 'gdppcg'
            combinedpdw = combinedpdw.merge(gdptabg,on=['time',aggunit])
        else:
            combinedpdw['gdppcg'] = np.log(combinedpdw['gdppc'])

        combinedpd = combinedpdw

    else:
        if growth:
            gdptab  = np.log(combinedpd_orig.reset_index().pivot(
                            index='time',columns=aggunit,values='gdppc'))
            gdptabg = gdptab.diff(1).stack()
            gdptabg.name = 'gdppcg'
            combinedpd = combinedpd.drop(columns='gdppcg').merge(gdptabg,on=['time',aggunit])

    if growth:
        adf = '_growth'
    else:
        adf = ''

    if anoms:
        fname = '../data/gdp_fits/get_fe_ds_{}{}_anom.csv'.format(aggunit,adf)

        combinedpd       = combinedpd.reset_index()
        combinedpd_means = combinedpd.groupby(aggunit)[['T','P']].mean()
        combinedpd_means = combinedpd_means.rename(columns={'T':'Tmean','P':'Pmean'})
        combinedpd       = combinedpd.merge(combinedpd_means.reset_index(),on=aggunit)
        combinedpd       = combinedpd.rename(columns={'T':'Tabs','P':'Pabs'})
        combinedpd['T']  = combinedpd['Tabs'] - combinedpd['Tmean']
        combinedpd['P']  = combinedpd['Pabs'] - combinedpd['Pmean']

        combinedpd['T2'] = combinedpd['T']**2
        combinedpd['P2'] = combinedpd['P']**2
        combinedpd['Tabs2'] = combinedpd['Tabs']**2
        combinedpd['Pabs2'] = combinedpd['Pabs']**2
        combinedpd['Tmean2'] = combinedpd['Tmean']**2
        combinedpd['Pmean2'] = combinedpd['Pmean']**2

    else:
        fname = '../data/gdp_fits/get_fe_ds_{}{}.csv'.format(aggunit,adf)

    combinedpd.to_csv(fname) 
    print(fname)
    formula_fe = 'gdppcg~T+T2+P+P2 | {}+ time'.format(aggunit)
    if 'pixel' in aggunit:
        subprocess.call(['./fit_and_project_fe.r',fname,formula_fe,'pop'])
    else:
        subprocess.call(['./fit_and_project_fe.r',fname,formula_fe,'None'])

    

