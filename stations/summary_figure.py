"""
This file makes Figure 1.
It uses data from: 
    GHCN database (../data/ghcnd/)
    shapefiles from Natural earth (../data/naturalearth/)
    results from monthly anomaly data from stations, grouped by political boundaries 
        (found in ../data/corout/monthly_anom)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import geopandas as gpd
import pandas as pd
from cartopy import crs as ccrs
import os
import random
from shapely.geometry import Point
from scipy.optimize import curve_fit
from calendar import monthrange
import matplotlib.colors as mcol
from datetime import datetime
import xarray as xr
import pyproj
aeqd = pyproj.Proj(proj='aeqd', ellps='WGS84', datum='WGS84', lat_0=0, lon_0=0).srs
import pdb

#   Read in saved data
coords = gpd.read_file('../data/ghcnd/stationcoords.shp')
headers = pd.read_csv('../data/ghcnd/stationheaders.csv',index_col=0)

#   Make maps of all T and P measurements colored by the last date of availability
headersT = headers[headers[['TMAX','TAVG']].sum(1)>0]
headersP = headers[headers['PRCP']>0]

coordsT = coords[coords['station'].isin(headersT.index)]
coordsP = coords[coords['station'].isin(headersP.index)]

yrs = [c for c in coords.columns if (c.startswith('1') or c.startswith('2'))]

#   Reverse cumulative sum of years with data
#   That way the first 1 will be the last year of data
datayrsR = np.cumsum(coords[yrs[::-1]].astype(int),1)

meltedyrs = datayrsR.join(coords['station']).melt(id_vars='station')
meltedyrs = meltedyrs.rename(columns={'variable':'latestyr'})

meltedyrs1 = meltedyrs[meltedyrs['value']==1]

meltedavail = coords[['station']+yrs].melt(id_vars='station')
meltedavail = meltedavail.rename(columns={'value':'avail','variable':'latestyr'})

meltedyrs1 = meltedyrs1.merge(meltedavail,'left',on=['station','latestyr'])
meltedyrs1 = meltedyrs1[meltedyrs1['avail']=='1']

coordsT = coordsT.merge(meltedyrs1[['station','latestyr']],'inner')
coordsP = coordsP.merge(meltedyrs1[['station','latestyr']],'inner')

coordsT['latestyr'] = coordsT['latestyr'].astype(int)
coordsP['latestyr'] = coordsP['latestyr'].astype(int)


#   Plot (a) and (b)
robinson = ccrs.Robinson().proj4_init
coordsT = coordsT.to_crs(robinson)
coordsP = coordsP.to_crs(robinson)

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

countries = gpd.read_file('../data/naturalearth/ne_50m_admin_0_countries.shp')
countriesr = countries.to_crs(robinson)

cmap = truncate_colormap(plt.cm.copper,minval=0.1,maxval=1,n=72)
ms = 0.5
plt.figure(figsize=(12,8))
ax1 = plt.subplot(3,2,1)
gs = 0.65
countriesr.plot(ax=ax1,color=[gs]*3)
countriesr.boundary.plot(ax=ax1,color='k',linewidth=0.5)
Ttitle =  'Year of most recent T measurement'
coordsT.sort_values('latestyr').plot('latestyr',ax=ax1,cmap=cmap,markersize=ms,vmin=1980,
                                     legend=True,legend_kwds={'extend':'min'})
ax1.set_title(Ttitle)
ax1.set_ylim([-6e6,ax1.get_ylim()[1]])
ax1.axis('off')
ax1.annotate('(a)',(0.15,0),xycoords='axes fraction',fontsize=14)

ax2 = plt.subplot(3,2,3)
countriesr.plot(ax=ax2,color=[gs]*3)
countriesr.boundary.plot(ax=ax2,color='k',linewidth=0.5)
Ptitle =  'Year of most recent P measurement'
coordsP.sort_values('latestyr').plot('latestyr',ax=ax2,cmap=cmap,markersize=ms,vmin=1980,
                                     legend=True,legend_kwds={'extend':'min'})
ax2.set_title(Ptitle)
ax2.set_ylim([-6e6,ax2.get_ylim()[1]])
ax2.axis('off')
ax2.annotate('(b)',(0.15,0),xycoords='axes fraction',fontsize=14)

#   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   

def reorient_netcdf(fp):
    """  
    Function to orient netcdf wrt -180,180 longitude (modified from Jacob)
    """
    f = xr.open_dataset(fp)
    if np.max(f.coords['lon'] > 180):
        new_lon = [-360.00 + num if num > 180 else num for num in f.coords['lon'].values]
        f = f.assign_coords({'lon':new_lon})
        f = f.sortby(f.coords['lon'])
    return f

udelT = reorient_netcdf('../data/UDEL/air.mon.mean.v501.nc')

grid0 = udelT.isel({'time':0}).to_dataframe()
grid = grid0.loc[~np.isnan(grid0['air'])]
grid = grid.reset_index()[['lat','lon']]
pts = [Point(pt['lon'],pt['lat']) for i,pt in grid.iterrows()]
grid['geometry'] = pts
grid = gpd.GeoDataFrame(grid)
grid = grid.set_crs("EPSG:4326")
grid = grid.to_crs(coordsP.crs)

#   Find nearest station to grid centers
gridjoin = gpd.sjoin_nearest(grid,coordsP[coordsP['2021']=='1'],distance_col='distance')
gridjoin = gridjoin.rename(columns={'lat_left':'lat','lon_left':'lon'})
gridjoin = gridjoin[['lat','lon','distance']].set_index(['lat','lon']).drop_duplicates()
gridjoinxr = gridjoin.to_xarray()
gridjoinxr = gridjoinxr.where(gridjoinxr.lat>-60,drop=True)

#   Plot (c)
axcarto = plt.subplot(3,2,5,projection=ccrs.Robinson(),frameon=False)
cm = axcarto.pcolormesh(gridjoinxr.lon,gridjoinxr.lat,gridjoinxr['distance']/1e3, 
                        transform=ccrs.PlateCarree(),vmin=0,vmax=100,cmap=plt.cm.Reds)
axcarto.coastlines()
plt.colorbar(cm,extend='max')
axcarto.set_title('Distance to nearest 2021 P measurement')
axcarto.annotate('(c)',(0.15,0),xycoords='axes fraction',fontsize=14)

#   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   

agtime = 'monthly_anom'
Tvar = 'TAVG'

def read_corout(name,Tvar):
    """
    Function to read data grouped by geographic area
    """
    corout = {}
    for n in os.listdir('../data/corout/{}'.format(name)):
        npath = '../data/corout/{}/{}/'.format(name,n)
        fs = os.listdir(npath)
        fs = [f for f in fs if Tvar in f]
        if len(fs)==0:
            fs = [f for f in fs if '_' not in f]
    
        if len(fs)>0:
            corout[n] = {}
            for f in fs:
                try:
                    tmpdat = pd.read_csv(npath+f)
                    if not name=='UDEL':
                        tmpdat = tmpdat.set_index('STATION',drop=True)
                    corout[n][f.split('.')[0].split('_')[0]] = tmpdat
                except:
                    print('Skipping {}'.format(n))    

    return corout

corout = read_corout(agtime,Tvar)
regions = list(corout.keys())

#   Create combo shapefiles with countries and admin areas where available (for plotting)
countries= countries.rename(columns={'NAME':'name','ISO_A2':'iso_a2'})
countries['adm1_code'] = countries['ISO_A3']+'-0000'
admin = gpd.read_file('../data/naturalearth/ne_50m_admin_1_states_provinces_lakes.shp')
combined = []
for ctry in countries['iso_a2'].unique():
    if ctry in admin['iso_a2'].values:
        entry = admin[admin['iso_a2']==ctry][['iso_a2','name','adm1_code','geometry']]
    else:
        entry = countries[countries['iso_a2']==ctry][['iso_a2','adm1_code','name','geometry']]
    combined.append(entry)
combined = gpd.GeoDataFrame(pd.concat(combined)).reset_index()

#   As a reference, calculate scales of countries

def fitf(x,m,c):
    return np.exp(-m*x)+c

def SE(y,yfit,x):
    n = len(y)
    se = np.sqrt(1/(n-2) * np.sum((y-yfit)**2)/np.sum((x-np.mean(x))**2))
    return se

def expreg(corout,regions,combined,limitN=False):
    regout  = {}
    for reg in regions:
        regout[reg] = {}
        item = corout[reg]
        distance = item['distance']
        precipcor = item['precipcor']
        tempcor = item['tempcor']
        if limitN:
            NcorT = item['NcorT']
            NcorP = item['NcorP']
        maskup = np.triu(np.ones(distance.shape),1)==1
    
        try:
            yp= precipcor.values[maskup].astype(float)
            yt= tempcor.values[maskup].astype(float)
        except:
            pdb.set_trace()
    
        x = (distance.values[maskup]/1000).astype(float)

        if limitN:
            nnip = ~np.isnan(yp) & (NcorP.values[maskup]>6)
            nnit = ~np.isnan(yt) & (NcorT.values[maskup]>6)
        else:
            nnip = ~np.isnan(yp)
            nnit = ~np.isnan(yt)

        yp = yp[nnip]
        xp = x[nnip]
        yt = yt[nnit]
        xt = x[nnit]
    
        if (len(yp)>2) & (len(yt)>2):
            poptP, pcovP = curve_fit(fitf,xp,yp,p0=[0.003,1])
            efoldP = 1/poptP[0]
            poptT, pcovT = curve_fit(fitf,xt,yt,p0=[0.0003,1])
            efoldT = 1/poptT[0]
            SEp = SE(yp,fitf(xp,*poptP),xp)
            SEt = SE(yt,fitf(xt,*poptT),xt)
    
            regout[reg]['P_coef'] = poptP[0]
            regout[reg]['P_efold'] = efoldP
            regout[reg]['P_SE'] = SEp
            regout[reg]['T_coef'] = poptT[0]
            regout[reg]['T_efold'] = efoldT
            regout[reg]['T_SE'] = SEt
            if len(poptP)>0:
                regout[reg]['P_offset'] = poptP[1]
                regout[reg]['T_offset'] = poptT[1]
            else:
                regout[reg]['P_offset'] = 0
                regout[reg]['T_offset'] = 0
    
    regout = pd.DataFrame(regout).T
    regout = regout.reset_index().rename(columns={'index':'adm1_code'})
    return regout

regout = expreg(corout,regions,combined)
combined2 = combined.merge(regout,'left',on='adm1_code')
combined2.loc[combined2['P_SE']/combined2['P_coef']>0.5,'P_efold'] = np.nan
combined2.loc[combined2['T_SE']/combined2['T_coef']>0.5,'T_efold'] = np.nan

#   Choose name of regions to plot
regs = ['BRA','CAN','IND']

def add_ref(scale):
    plt.plot([scale,scale],[-0.5,1],'--',color=[0.6]*3)

#   Read GADM shapefiles for admin unit scales,
#       since natural earth doesn't have the same administrative units
admingadm = gpd.read_file('../data/gadm/gadm36_1_lowres.shp')
scales = {}
for reg in regs:
    adminreg = admingadm[admingadm['GID_0']==reg[:3]]
    adminreg = adminreg.to_crs("+proj=eck4 +lon_0=0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs")
    scale = np.mean(np.sqrt((adminreg.bounds['maxx']-adminreg.bounds['minx'])**2 + 
                            (adminreg.bounds['maxy']-adminreg.bounds['miny'])**2)/1e3)
    scales[reg[:3]] = int(np.floor(scale))

Tploti = [3,7,11]
Pploti = [4,8,12]
abc1 = ['d','e','f']
abc2 = ['g','h','i']
for ri,reg in enumerate(regs):
    item = corout[reg]
    distance = item['distance']
    NcorP = item['NcorP']
    NcorT = item['NcorT']
    precipcor = item['precipcor']
    tempcor = item['tempcor']

    name = admingadm[admingadm['GID_1'].str.startswith(reg[:3])]['NAME_0'].values[0]
    regreg = regout.loc[regout['adm1_code']==reg].iloc[0]

    #   Plot as a function of distance between points
    maskup = np.triu(np.ones(distance.shape))==1
    NcorPup = NcorP.values[maskup]
    NcorTup = NcorT.values[maskup]

    plt.subplot(3,4,Tploti[ri])
    plt.cla()
    distvar = np.abs(distance.copy())
    x = (distvar.values[maskup]/1000).astype(float)

    #   Limit to correlations that are based on at least 6 data points
    minNiP = NcorPup>6
    minNiT = NcorTup>6

    plt.scatter(x[minNiP],precipcor.values[maskup][minNiP],1,c=NcorPup[minNiP],cmap=plt.cm.Blues,vmin=0,vmax=150)
    plt.plot(np.sort(x),fitf(np.sort(x),regreg['P_coef'],regreg['P_offset']),color='k')
    if ri==2:
        plt.xlabel('Distance between stations [km]',fontsize=11)
    elif ri==0:
        plt.title('Precipitation')
    plt.ylabel('Station correlation coef.',fontsize=11)
    plt.ylim([-1,1])
    add_ref(scales[reg[:3]])
    Pfoldstr = 'e-folding: {:.0f}km'.format(regreg['P_efold'])
    plt.text(0,-0.9,'{}\n{}'.format(name,Pfoldstr))
    plt.colorbar(extend='max')
    plt.annotate('({})'.format(abc1[ri]),(0.83,0.05),xycoords='axes fraction',fontsize=14)

    plt.subplot(3,4,Pploti[ri])
    plt.scatter(x[minNiT],tempcor.values[maskup][minNiT],1,c=NcorTup[minNiT],cmap=plt.cm.Blues,vmin=0,vmax=150)
    plt.plot(np.sort(x),fitf(np.sort(x),regreg['T_coef'],regreg['T_offset']),color='k')
    if ri==2:
        plt.xlabel('Distance between stations [km]',fontsize=11)
    elif ri==0:
        plt.title('Temperature')
    plt.ylim([-1,1])
    add_ref(scales[reg[:3]])
    Tfoldstr = 'e-folding: {:.0f}km'.format(regreg['T_efold'])
    plt.text(0,-0.9,'{}\n{}'.format(name,Tfoldstr))
    plt.colorbar(extend='max')
    plt.annotate('({})'.format(abc2[ri]),(0.83,0.05),xycoords='axes fraction',fontsize=14)

plt.tight_layout()

plt.savefig('../figures/figure1.png',dpi=600)
#   plt.savefig('Fig1_scales_summary.pdf')

