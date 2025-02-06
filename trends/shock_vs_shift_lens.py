"""
This file makes Figure 7.
"""

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cfeature

Tslopes = xr.open_dataarray('../data/CMIP6/CESM2/tas_slopes_lens05.nc')
Tshocks = xr.open_dataarray('../data/CMIP6/CESM2/tas_shocks_lens05.nc')
Tlensm  = xr.open_dataset('../data/CMIP6/CESM2/LENS2_TREFHT_PRECT_05deg_ensmean_1960_2050.nc')
Tlensm  = Tlensm['TREFHT']

Tratio = Tslopes / Tshocks

weights = np.cos(np.deg2rad(Tslopes.lat))
weights.name = "weights"
Tslopesmean = Tslopes.weighted(weights).mean(['lat','lon'])
Tshocksmean = Tshocks.weighted(weights).mean(['lat','lon'])
Tmean       = Tlensm.weighted(weights).mean(['lat','lon'])

Tratiomean = Tratio.weighted(weights).mean(['lat','lon'])

combxr = xr.open_dataset('../data/gdp/combxr_05deg.nc')
gdppc = combxr['gdppc'].isel({'time':-1})

Tratio_1970s = Tratio.sel(year=range(1970,1980)).mean('year').transpose()
Tratio_2040s = Tratio.sel(year=range(2040,2050)).mean('year').transpose()

Tratio_1970s_land = Tratio_1970s.where(gdppc>0)
Tratio_2040s_land = Tratio_2040s.where(gdppc>0)

def plot_map(subplot,dat,letter,vmin,vmax,cmap,cbarlab=''):
    ax = plt.subplot(*subplot,projection=ccrs.Robinson())
    im = dat.plot(ax=ax,cmap=cmap,vmin=vmin,vmax=vmax,transform=ccrs.PlateCarree(),
                  add_colorbar=False)
    cb = plt.colorbar(im,ax=ax,fraction=0.024,pad=0.04,extend='max')
    cb.set_label(label=cbarlab,rotation=270,va='bottom',fontsize=16)
                     
    ax.axis('off')
    ax.coastlines(lw=0.5)
    ax.set_ylim((-6025154.6651, 8625154.6651))
    ax.annotate('({})'.format(letter),(0.13,0.05),xycoords='axes fraction',fontsize=13)
    ax.set_xlabel('')
    ax.set_ylabel('')
    return ax

plt.figure(figsize=(18,3))
lab1 = r'$\frac{^o\mathrm{C / decade}}{^o\mathrm{C}}$'
ax1 = plot_map([1,3,1],Tratio_1970s_land,'a',-0.3,0.3,'PuOr',lab1)
ax1.set_title('Decaldal trend/variability, 1970s mean')
ax2 = plot_map([1,3,2],Tratio_2040s_land,'b',-0.3,0.3,'PuOr',lab1)
ax2.set_title('Decaldal trend/variability, 2040s mean')
lab3 = r'$\Delta\frac{^o\mathrm{C / decade}}{^o\mathrm{C}}$'
ax3 = plot_map([1,3,3],Tratio_2040s_land-Tratio_1970s_land,'c',-0.3,0.3,'RdBu_r',lab3)
ax3.set_title('Decaldal trend/variability, 2040s-1970s')
plt.tight_layout()

plt.savefig('../figures/figure7.png',dpi=600)

