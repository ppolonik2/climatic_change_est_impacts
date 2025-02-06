import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import geopandas as gpd
import os
import pdb

#   Take gridded GDP (with projection), gridded dT (with projection) and regression coefficients
#   Return projected GDP with/without climate change
#       at whatever level (pixel, admin_id, country_id)

#   This section produces the maps that are plotted below
#   It is set to False because it takes to to run and the output is already saved in ../data/
if False:
    ssp = xr.open_dataset('~/CERM/data/input/GDP_pop_corrected.nc').sel(ssp=2)
    dssp = (ssp['GDPpc'] / ssp['GDPpc'].shift(year=1)) - 1
    dssp.name = 'gdppcg'
    
    T = xr.open_dataset('../data/CMIP6/CESM2/tas.ensmean.ssp245.annual.nc')
    P = xr.open_dataset('../data/CMIP6/CESM2/pr.ensmean.ssp245.annual.nc') * 3.154e6
    combxr = xr.open_dataset('../data/gdp/combxr_05deg2.nc')
    combxr = combxr.rename({'time':'year'})
    
    Tbias = T     .sel(year=range(1991,2015)).mean('year')['tas'] - \
            combxr.sel(year=range(1991,2015)).mean('year')['T']
            
    Pbias = P     .sel(year=range(1991,2015)).mean('year')['pr'] - \
            combxr.sel(year=range(1991,2015)).mean('year')['P']
    
    T = T['tas'] - Tbias
    P = P['pr' ] - Pbias
    P = P.where(P>0,0)
    T.name = 'T'
    P.name = 'P'
    
    Ttot=xr.concat([combxr['T'].sel(year=range(1900,2016)),T.sel(year=range(2016,2100))],dim='year')
    Ptot=xr.concat([combxr['P'].sel(year=range(1900,2016)),P.sel(year=range(2016,2100))],dim='year')

    combxr = xr.merge([dssp,ssp[['gdp','GDPpc','pop']],combxr[['country_id','admin_id']],T,P])
    combxr = combxr.drop(['T','P'])
    combxr = xr.merge([combxr,Ttot,Ptot])
    
    def agg(dat,cid,pop=[],sumOnly=False):
        """
        take gridded data, country id, and population (optional)
        return country-level data, aggregated by id
        """
        if len(pop)>0:
            popsum   = pop.groupby(cid).sum()
            weighted = (dat*pop).groupby(cid).sum()
            if sumOnly:
                out = weighted
            else:
                out = weighted/popsum
    
        else:
            if sumOnly:
                out = dat.groupby(cid).sum()
            else:
                out = dat.groupby(cid).mean()
    
        return out
    
    def unagg(combxrag,combxr,agunit):
        """
        Take aggregated and make it pixel level for plotting - constant at each grid cell
        """
        #   Initialize
        combxrunag = combxr.copy()
        for v in list(combxrag):
            combxrunag[v] = combxr['GDPpc'].copy()
        combxrunag = combxrunag[list(combxrag)]
    
        #   Run through each agunit and assign corresponding constant value to each
        for au in combxrag[agunit].values:
            print(au)
            combxrunag = xr.where(combxr[agunit]==au,combxrag.sel({agunit:au}),combxrunag)
        return combxrunag
    
    
    def project(combxr,coefs,agunit):
        if agunit=='pixel':
            combxrag = combxr.copy()
        else:
            agclim = agg(combxr[['T','P']],combxr[agunit],combxr['pop'],sumOnly=False)
    
            #   GDPpc * pop to get aggregated GDP
            aggdp  = agg(combxr[['gdp']],combxr[agunit],sumOnly=True)
    
            #   Then divide by aggregated pop to get GDPpc back
            agpop  = agg(combxr[['pop']],combxr[agunit],sumOnly=True)
    
            aggdppc = aggdp['gdp'] / agpop['pop']
            aggdppc.name = 'GDPpc'
    
            aggdppcg = (aggdppc / aggdppc.shift(year=1)) - 1
            aggdppcg.name = 'gdppcg'
    
            combxrag = xr.merge([agclim,aggdppc,aggdppcg,agpop])
    
        
        # Put actual projection from below in here
        T10 = combxrag['T'].sel(year=range(2010,2015)).mean('year')
        T   = combxrag['T']
        P10 = combxrag['P'].sel(year=range(2010,2015)).mean('year')
        P   = combxrag['P']
    
        base_T = coefs.loc['T','coef'] * T10 + coefs.loc['T2','coef'] * T10*T10
        base_P = coefs.loc['P','coef'] * P10 + coefs.loc['P2','coef'] * P10*P10
    
        proj_T = coefs.loc['T','coef'] * T   + coefs.loc['T2','coef'] * T  *T
        proj_P = coefs.loc['P','coef'] * P   + coefs.loc['P2','coef'] * P  *P
    
        base = base_T + base_P
        proj = proj_T + proj_P
        
        d_T = proj_T - base_T
        d_P = proj_P - base_P
        d   = proj   - base
        
        #   prev_gdppc * (1 + base_growth + d)
        combxrag['GDPpc_clim_T'] = combxrag['GDPpc'].copy()
        combxrag['GDPpc_clim_P'] = combxrag['GDPpc'].copy()
        combxrag['GDPpc_clim']   = combxrag['GDPpc'].copy()
        combxrag['GDPpc_clim_test']   = combxrag['GDPpc'].copy()
        for y in range(2016,2100):
            oldGDP_T = combxrag['GDPpc_clim_T'].sel(year=y-1).drop('year')
            oldGDP_P = combxrag['GDPpc_clim_P'].sel(year=y-1).drop('year')
            oldGDP   = combxrag['GDPpc_clim'].sel(year=y-1).drop('year')
            oldGDP_test   = combxrag['GDPpc_clim_test'].sel(year=y-1).drop('year')
            change_T = (1 + combxrag['gdppcg'].sel(year=y).drop('year') + d_T.sel(year=y).drop('year') ) 
            change_P = (1 + combxrag['gdppcg'].sel(year=y).drop('year') + d_P.sel(year=y).drop('year') ) 
            change   = (1 + combxrag['gdppcg'].sel(year=y).drop('year') + d.sel(year=y).drop('year') ) 
            change_test   = (1 + combxrag['gdppcg'].sel(year=y).drop('year') + 0 ) 
            combxrag['GDPpc_clim_T'].loc[{'year':y}] = oldGDP_T * change_T
            combxrag['GDPpc_clim_P'].loc[{'year':y}] = oldGDP_P * change_P
            combxrag['GDPpc_clim'].loc[{'year':y}]   = oldGDP   * change
            combxrag['GDPpc_clim_test'].loc[{'year':y}]   = oldGDP_test   * change_test
    
        gdpvs = ['GDPpc','GDPpc_clim_T','GDPpc_clim_P','GDPpc_clim','GDPpc_clim_test']
    
        if agunit=='admin_id':
            #   aggregate from admin to country, then unaggregate
            #   Construct mapping and add to dataset as coordinate
            mapdict = combxr[['country_id','admin_id']].to_dataframe().dropna(
                        ).reset_index()[['admin_id','country_id']].drop_duplicates(
                        ).set_index('admin_id').to_dict()['country_id']
            ctrymap = [mapdict[a] for a in combxrag.admin_id.values]
            combxrag = combxrag.assign_coords({'country_id':('admin_id',ctrymap)})
            combxrag_pop = combxrag['pop'].groupby('country_id').sum()
    
        elif agunit=='pixel':
            combxrag_pop = combxrag[['country_id','pop']].groupby('country_id').sum()['pop']
    
        if agunit in ['pixel','admin_id']:
            #   Get GDP
            gdpvars = []
            for v in gdpvs:
                combxrag_gdp = (combxrag[v]*combxrag['pop']).groupby(combxrag['country_id']).sum()
                newgdppc = combxrag_gdp/combxrag_pop
                newgdppc.name = v
                gdpvars.append(newgdppc)
            out = xr.merge(gdpvars+[combxrag_pop])
    
        else:
            out = combxrag
            
        combxrunag = unagg(out,combxr,agunit='country_id')
    
        return combxrag, combxrunag
         
    # Run projection for pixel, admin, and country and compare country-level maps
    agunits = ['pixel','admin_id','country_id']
    coefs_all = {}
    for agunit in agunits:
        coefstmp = pd.read_csv('../data/gdp_fits/get_fe_ds_{}_growth_fit.csv'.format(agunit),index_col=0)
        coefs_all[agunit] = coefstmp
    
    corfile = '../data/gdp_fits/spreadcorrect_coefs_{}_L-1.00_G1.00_growth.csv'.format('pixel')
    coefscorrect = pd.read_csv(corfile,index_col=0)
    coefscorrect = coefscorrect.rename(columns={'coef_corrected':'coef'})
    
    combxrag_pixel,   combxrunag_pixel   = project(combxr,coefs_all['pixel']     ,agunit='pixel')
    combxrag_pixel_c, combxrunag_pixel_c = project(combxr,coefscorrect           ,agunit='pixel')
    combxrag_admin,   combxrunag_admin   = project(combxr,coefs_all['admin_id']  ,agunit='admin_id')
    combxrag_country, combxrunag_country = project(combxr,coefs_all['country_id'],agunit='country_id')

    combxrag_pixel.to_netcdf('../data/gdp/projections/combxrag_pixel.nc')
    combxrunag_pixel.to_netcdf('../data/gdp/projections/combxrunag_pixel.nc')
    combxrag_pixel_c.to_netcdf('../data/gdp/projections/combxrag_pixel_c.nc')
    combxrunag_pixel_c.to_netcdf('../data/gdp/projections/combxrunag_pixel_c.nc')
    combxrag_admin.to_netcdf('../data/gdp/projections/combxrag_admin.nc')
    combxrunag_admin.to_netcdf('../data/gdp/projections/combxrunag_admin.nc')
    combxrag_country.to_netcdf('../data/gdp/projections/combxrag_country.nc')
    combxrunag_country.to_netcdf('../data/gdp/projections/combxrunag_country.nc')

else:
    combxrag_pixel     = xr.open_dataset('../data/gdp/projections/combxrag_pixel.nc')
    combxrunag_pixel   = xr.open_dataset('../data/gdp/projections/combxrunag_pixel.nc')
    combxrag_pixel_c   = xr.open_dataset('../data/gdp/projections/combxrag_pixel_c.nc')
    combxrunag_pixel_c = xr.open_dataset('../data/gdp/projections/combxrunag_pixel_c.nc')
    combxrag_admin     = xr.open_dataset('../data/gdp/projections/combxrag_admin.nc')
    combxrunag_admin   = xr.open_dataset('../data/gdp/projections/combxrunag_admin.nc')
    combxrag_country   = xr.open_dataset('../data/gdp/projections/combxrag_country.nc')
    combxrunag_country = xr.open_dataset('../data/gdp/projections/combxrunag_country.nc')


#   Figures
#   One of pixel with/without correction
#   One of country-level projection with each 
yr = range(2045,2055)

#   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   
#   Begin figure section
#   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   

ctryshp = gpd.read_file('../data/gadm/gadm36_0_lowlowres.shp')
ctryshp = ctryshp.to_crs(ccrs.Robinson())
meta    = pd.read_csv('../data/geo/country_id_mapping.csv',index_col=0)
meta.columns = ['GID_0','country_id']

#   Create country shapefiles from datasets for chloropleths
combxrag_admin['GDP_clim'] = combxrag_admin['GDPpc_clim'] * combxrag_admin['pop']
combxrag_admin['GDP']      = combxrag_admin['GDPpc']      * combxrag_admin['pop']
combxrag_admin_ctry = combxrag_admin.groupby('country_id').sum()
combxrag_admin_ctry['GDPpc_clim'] = combxrag_admin_ctry['GDP_clim'] / combxrag_admin_ctry['pop']
combxrag_admin_ctry['GDPpc']      = combxrag_admin_ctry['GDP']      / combxrag_admin_ctry['pop']

#   damaged
combxrag_admin_ctry_clim = combxrag_admin_ctry['GDPpc_clim'].to_dataset('year').drop('ssp').to_dataframe()
combxrag_admin_ctry_clim = combxrag_admin_ctry_clim.reset_index().merge(meta)
combxrag_admin_ctry_clim = ctryshp.merge(combxrag_admin_ctry_clim)
combxrag_admin_ctry_clim = combxrag_admin_ctry_clim[['GID_0']+list(yr)+['geometry']]

#   undamaged
combxrag_admin_ctry_noclim = combxrag_admin_ctry['GDPpc'].to_dataset('year').drop('ssp').to_dataframe()
combxrag_admin_ctry_noclim = combxrag_admin_ctry_noclim.reset_index().merge(meta)
combxrag_admin_ctry_noclim = ctryshp.merge(combxrag_admin_ctry_noclim)

#   merge
combxrag_admin_ctry_clim = combxrag_admin_ctry_clim[['GID_0']+list(yr)+['geometry']]
combxrag_admin_ctry_clim['GDPpc_clim'] = combxrag_admin_ctry_clim[list(yr)].mean(1)
combxrag_admin_ctry_clim['GDPpc']      = combxrag_admin_ctry_noclim[list(yr)].mean(1)
combxrag_admin_ctry_clim['ratio']      = (combxrag_admin_ctry_clim[list(yr)]/combxrag_admin_ctry_noclim[list(yr)]).mean(1)
combxrag_admin_ctry_clim['ratio1_admin'] = (combxrag_admin_ctry_clim['ratio'] - 1)*100

#   Now repeat basically the same thing for country
combxrag_country['GDP_clim'] = combxrag_country['GDPpc_clim'] * combxrag_country['pop']
combxrag_country['GDP']      = combxrag_country['GDPpc']      * combxrag_country['pop']
combxrag_country_clim = combxrag_country['GDPpc_clim'].to_dataset('year').drop('ssp').to_dataframe()
combxrag_country_clim = combxrag_country_clim.reset_index().merge(meta)
combxrag_country_clim = ctryshp.merge(combxrag_country_clim)
combxrag_country_clim = combxrag_country_clim[['GID_0']+list(yr)+['geometry']]

#   undamaged
combxrag_country_noclim = combxrag_country['GDPpc'].to_dataset('year').drop('ssp').to_dataframe()
combxrag_country_noclim = combxrag_country_noclim.reset_index().merge(meta)
combxrag_country_noclim = ctryshp.merge(combxrag_country_noclim)

#   merge
combxrag_country_clim = combxrag_country_clim[['GID_0']+list(yr)+['geometry']]
combxrag_country_clim['GDPpc_clim'] =  combxrag_country_clim[list(yr)].mean(1)
combxrag_country_clim['GDPpc']      =  combxrag_country_noclim[list(yr)].mean(1)
combxrag_country_clim['ratio']      = (combxrag_country_clim[list(yr)]/combxrag_country_noclim[list(yr)]).mean(1)
combxrag_country_clim['ratio1_country'] = (combxrag_country_clim['ratio'] - 1)*100

#   And the same thing again for pixel, but format is pretty different
combxrag_pixel['GDP_clim'] = combxrag_pixel['GDPpc_clim'] * combxrag_pixel['pop']
combxrag_pixel['GDP']      = combxrag_pixel['GDPpc']      * combxrag_pixel['pop']

combxrag_pixel_ctry = combxrag_pixel[['GDP','GDP_clim','pop','country_id']].groupby('country_id').sum()
combxrag_pixel_ctry['GDPpc']      = combxrag_pixel_ctry['GDP']     /combxrag_pixel_ctry['pop']
combxrag_pixel_ctry['GDPpc_clim'] = combxrag_pixel_ctry['GDP_clim']/combxrag_pixel_ctry['pop']

#   damaged
combxrag_pixel_ctry_clim = combxrag_pixel_ctry['GDPpc_clim'].to_dataset('year').drop('ssp').to_dataframe()
combxrag_pixel_ctry_clim = combxrag_pixel_ctry_clim.reset_index().merge(meta)
combxrag_pixel_ctry_clim = ctryshp.merge(combxrag_pixel_ctry_clim)
combxrag_pixel_ctry_clim = combxrag_pixel_ctry_clim[['GID_0']+list(yr)+['geometry']]

#   undamaged
combxrag_pixel_ctry_noclim = combxrag_pixel_ctry['GDPpc'].to_dataset('year').drop('ssp').to_dataframe()
combxrag_pixel_ctry_noclim = combxrag_pixel_ctry_noclim.reset_index().merge(meta)
combxrag_pixel_ctry_noclim = ctryshp.merge(combxrag_pixel_ctry_noclim)

#   merge
combxrag_pixel_ctry_clim = combxrag_pixel_ctry_clim[['GID_0']+list(yr)+['geometry']]
combxrag_pixel_ctry_clim['GDPpc_clim'] = combxrag_pixel_ctry_clim[list(yr)].mean(1)
combxrag_pixel_ctry_clim['GDPpc']      = combxrag_pixel_ctry_noclim[list(yr)].mean(1)
combxrag_pixel_ctry_clim['ratio']      = (combxrag_pixel_ctry_clim[list(yr)]/combxrag_pixel_ctry_noclim[list(yr)]).mean(1)
combxrag_pixel_ctry_clim['ratio1_pixel'] = (combxrag_pixel_ctry_clim['ratio'] - 1)*100

#   Combine into one dataframe
all3 = combxrag_country_clim[['GID_0','ratio1_country','geometry']]
all3 = all3.merge(combxrag_admin_ctry_clim[['GID_0','ratio1_admin']])
all3 = all3.merge(combxrag_pixel_ctry_clim[['GID_0','ratio1_pixel']])
all3 = all3.dropna()

#   New chloropleth
fig = plt.figure(figsize=(6,8));
gmin, gmax = -50,50
ax1 = plt.subplot(3,1,1,projection=ccrs.Robinson())
all3.plot('ratio1_pixel',ax=ax1,vmin=gmin,vmax=gmax,cmap='RdBu',
            legend=True,legend_kwds={'extend':'both'})
all3.boundary.plot(ax=ax1,color='k',lw=0.2)
ax1.set_title('Pixel')
ax1.annotate('(a)',(0.01,0.98),xycoords='axes fraction',ha='left',va='top',fontsize=14)

ax2 = plt.subplot(3,1,2,projection=ccrs.Robinson())
all3.plot('ratio1_admin',ax=ax2,vmin=gmin,vmax=gmax,cmap='RdBu',
            legend=True,legend_kwds={'extend':'both'})
all3.boundary.plot(ax=ax2,color='k',lw=0.2)
ax2.set_title('Admin')
ax2.annotate('(b)',(0.01,0.98),xycoords='axes fraction',ha='left',va='top',fontsize=14)

ax3 = plt.subplot(3,1,3,projection=ccrs.Robinson())
all3.plot('ratio1_country',ax=ax3,vmin=gmin,vmax=gmax,cmap='RdBu',
            legend=True,legend_kwds={'extend':'both'})
all3.boundary.plot(ax=ax3,color='k',lw=0.2)
ax3.set_title('Country')
ax3.annotate('(c)',(0.01,0.98),xycoords='axes fraction',ha='left',va='top',fontsize=14)

fig.get_children()[2].set_ylabel('Country GDP$_{pc}$ Change [%]',rotation=270,va='bottom')
fig.get_children()[4].set_ylabel('Country GDP$_{pc}$ Change [%]',rotation=270,va='bottom')
fig.get_children()[6].set_ylabel('Country GDP$_{pc}$ Change [%]',rotation=270,va='bottom')

#   Add in missing shapes
missingctry = [c for c in ctryshp['GID_0'].values if c not in all3['GID_0'].values]
ctryshp2    = ctryshp.set_index('GID_0')
missingshp  = ctryshp2.loc[missingctry]
missingshp  = missingshp.drop('ATA')
missingshp.plot(ax=ax1,color='lightgrey')
missingshp.boundary.plot(ax=ax1,color='k',lw=0.2)
missingshp.plot(ax=ax2,color='lightgrey')
missingshp.boundary.plot(ax=ax2,color='k',lw=0.2)
missingshp.plot(ax=ax3,color='lightgrey')
missingshp.boundary.plot(ax=ax3,color='k',lw=0.2)

plt.tight_layout()

plt.savefig('../figures/figure3.png',dpi=600)

#   #   #   #   #   #   #   #   #   #   #   #   #   #
#   Figure 6, correction figure
#   #   #   #   #   #   #   #   #   #   #   #   #   #

plt.figure(figsize=(11,5));
rmin, rmax = -40,40
ax1 = plt.subplot(2,2,1,projection=ccrs.Robinson())
ratio1 = ((combxrag_pixel.sel(year=yr)['GDPpc_clim']) / combxrag_pixel.sel(year=yr)['GDPpc']).mean('year') - 1
ratio1 *= 100
ratio1.name = 'GDP$_{pc}$ Change [%]'
cm1 = plt.cm.RdBu.copy()
cm1.set_bad(color='white')
im = ratio1.plot(ax=ax1,vmin=rmin,vmax=rmax,cmap=cm1,transform=ccrs.PlateCarree(),
            add_colorbar=False)
cbar = plt.colorbar(im,extend='both')
cbar.ax.set_ylabel(ratio1.name,rotation=270,va='bottom')
ax1.coastlines()
plt.annotate('(a)',(0.01,0.98),xycoords='axes fraction',ha='left',va='top',fontsize=13)

ax2 = plt.subplot(2,2,2,projection=ccrs.Robinson())
ratio2 = (combxrag_pixel_c.sel(year=yr)['GDPpc_clim'] / combxrag_pixel_c.sel(year=yr)['GDPpc']).mean('year') - 1
ratio2 *= 100
ratio2.name = 'GDP$_{pc}$ Change [%]'
im = ratio2.plot(ax=ax2,vmin=rmin,vmax=rmax,cmap=cm1,transform=ccrs.PlateCarree(),
            add_colorbar=False)
cbar = plt.colorbar(im,extend='both')
cbar.ax.set_ylabel(ratio2.name,rotation=270,va='bottom')
ax2.coastlines()
plt.annotate('(b)',(0.01,0.98),xycoords='axes fraction',ha='left',va='top',fontsize=13)

ax3 = plt.subplot(2,2,3,projection=ccrs.Robinson())
ratiod = (ratio2-ratio1)
ratiod.name = 'Dif. in GDP$_{pc}$ Change [%]'
cm3 = plt.cm.coolwarm_r.copy()
cm3.set_bad(color='white')
im = ratiod.plot(ax=ax3,vmin=-10,vmax=10,cmap=cm3,transform=ccrs.PlateCarree(),
            add_colorbar=False)
cbar = plt.colorbar(im,extend='both')
cbar.ax.set_ylabel(ratiod.name,rotation=270,va='bottom')
ax3.coastlines()
plt.annotate('(c)',(0.01,0.98),xycoords='axes fraction',ha='left',va='top',fontsize=13)

ax4 = plt.subplot(2,2,4,projection=ccrs.Robinson())
ratio4_1 = ((combxrag_pixel.sel(year=yr)['GDPpc_clim_P']) / combxrag_pixel.sel(year=yr)['GDPpc']).mean('year') - 1
ratio4_2 = ((combxrag_pixel_c.sel(year=yr)['GDPpc_clim_P']) / combxrag_pixel.sel(year=yr)['GDPpc']).mean('year') - 1
ratio4 = ratio4_2 / ratio4_1
ratio4.name = 'Fractional change in P importance'
cm4 = plt.cm.Greens.copy()
cm4.set_bad(color='white')
im = ratio4.plot(ax=ax4,vmin=1.9,vmax=2.2,cmap=cm4,transform=ccrs.PlateCarree(),
            add_colorbar=False)
cbar = plt.colorbar(im,extend='both')
cbar.ax.set_ylabel(ratio4.name,rotation=270,va='bottom')
ax4.coastlines()
plt.annotate('(d)',(0.01,0.98),xycoords='axes fraction',ha='left',va='top',fontsize=13)

ax1.set_title('Uncorrected')
ax2.set_title('Corrected')
ax3.set_title('Difference')
ax4.set_title('P Change')
for ax in [ax1,ax2,ax3]:
    ax.set_xlabel('')
    ax.set_ylabel('')
plt.tight_layout()

plt.savefig('../figures/figure6.png',dpi=400)


