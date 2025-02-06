"""
This script makes Figure 2 from the CIRES standard deviations and idealized output
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec
import matplotlib as mpl
import cartopy.crs as ccrs
import seaborn as sns

exps = {
    'spread':{'file':'../data/idealized_fits/spread_fits.csv',
              'Perr':'1.0_spread', 
              'i':0,'line':'-','name':'Errors from spread'},
    'spreadrev':{'file':'../data/idealized_fits/spread_fits.csv',
                 'Perr':'1.0_spread_rev', 
                 'i':1,'line':'--','name':'Reversed spread errors'},
    'spreadshuf':{'file':'../data/idealized_fits/spread_fits.csv',
                  'Perr':'1.0_spread_shuf', 
                  'i':2,'line':':','name':'Shuffled errors'},
}

Tf = np.linspace(0,1,30)
Pf = np.linspace(0,1,30)

aggunits = ['pixel','admin_id','country_id']
fes = True
ensmean = True  #   If file has ensemble

setvals = {
    'cor(T,P)':'[-1.0, 1.0]',
    'outerr':'10_std',
    'interterm':0,
    'w_TP_true':10,
}

subtractTrue = False

agnamemap = {'pixel':'Pixel','admin_id':'Admin 1','country_id':'Country'}
abc = ['a','b','c','d','e','f','g','h']

#   Maps
Tratioan = xr.open_dataarray('../data/CIRES/err/Tstderr.nc')
Pratioan = xr.open_dataarray('../data/CIRES/err/Pstderr.nc')
Tratioan.name = 'T  uncertainty  [$\sigma_T$]'
Pratioan.name = 'P  uncertainty  [$\sigma_P$]'

Nwidth = 43
Nheight = 20
leftwidth = 20
topheight = 9
verbuf = 1
midbuf = 3
llbump = 2
shrinklower = 3

fig = plt.figure(figsize=(14,6))

#   Top row
ax1 = plt.subplot2grid((Nheight,Nwidth),(0,0),               topheight,leftwidth-1,fig=fig,
                        projection=ccrs.Robinson(),frameon=False)

ax2 = plt.subplot2grid((Nheight,Nwidth),(0,leftwidth+midbuf),topheight,leftwidth-1,fig=fig,
                       projection=ccrs.Robinson(),frameon=False)

cax1 = plt.subplot2grid((Nheight,Nwidth),(0,leftwidth-1),topheight,1,fig=fig)
cax2 = plt.subplot2grid((Nheight,Nwidth),(0,Nwidth-1)   ,topheight,1,fig=fig)

#   Bottom row
ax3 = plt.subplot2grid((Nheight,Nwidth),((topheight+verbuf),0+llbump)        ,topheight,leftwidth-shrinklower,fig=fig)
ax4 = plt.subplot2grid((Nheight,Nwidth),((topheight+verbuf),leftwidth+midbuf+llbump),topheight,leftwidth-shrinklower,fig=fig)

cmaper = cm.bone_r
combxr = xr.open_dataset('../data/gdp/combxr_05deg.nc')

im = Tratioan.plot(ax=ax1,transform=ccrs.PlateCarree(),vmin=0,vmax=2.5,
                   cmap=cmaper,add_colorbar=False)

cb = plt.colorbar(im,cax=cax1,shrink=0.5,extend='max')
cb.set_label(Tratioan.name,size=14,rotation=270,va='bottom')
cb.ax.tick_params(labelsize=14)
ax1.set_xlabel('')
ax1.set_ylabel('')
ax1.coastlines(lw=0.3)
ax1.annotate('(a)',(0.10,0.05),xycoords='axes fraction',fontsize=14)

im = Pratioan.plot(ax=ax2,transform=ccrs.PlateCarree(),vmin=0,vmax=2.5,
                   cmap=cmaper,add_colorbar=False)

cb = plt.colorbar(im,cax=cax2,shrink=0.5,extend='max')
cb.set_label(Pratioan.name,size=14,rotation=270,va='bottom')
cb.ax.tick_params(labelsize=14)
ax2.set_xlabel('')
ax2.set_ylabel('')
ax2.coastlines(lw=0.3)
ax2.annotate('(b)',(0.10,0.05),xycoords='axes fraction',fontsize=14)

for exp, info in exps.items():
    dat = pd.read_csv(info['file'])
    Truecoefs = dat[(np.abs(dat['w_PP']-1)<1e-5)&(dat['w_TP_true']==setvals['w_TP_true'])].iloc[0]

    #   Perr info sets P and T err (use same for both)
    dat = dat[(dat['Perr']==info['Perr'])&(dat['Terr']==info['Perr'])]
    
    fei = dat['feid']=='None'
    if fes:
        fei = ~fei
    
    subdat = dat[fei]
    
    for col, val in setvals.items():
        subdat = subdat[dat[col]==val]
    
    Tfp = Tf[:-1] + np.diff(Tf)
    Pfp = Pf[:-1] + np.diff(Pf)

    for agi, aggunit in enumerate(['pixel']):
        #   Plot the true curve, but only once
        if info['i']==0:
            Tcurvetrue = Tf*Truecoefs['T'] + Tf*Tf*Truecoefs['T2']
            Pcurvetrue = Pf*Truecoefs['P'] + Pf*Pf*Truecoefs['P2']

            #   Make separate variables so subtract for subtractTrue works 
            Tcurvetrued = np.diff(Tcurvetrue)
            Pcurvetrued = np.diff(Pcurvetrue)
            Tcurvetruep = Tcurvetrued.copy()
            Pcurvetruep = Pcurvetrued.copy()
            if subtractTrue:
                Tcurvetruep -= Tcurvetruep
                Pcurvetruep -= Pcurvetruep

        plotdat = subdat[subdat['aggunit']==aggunit]
        Tcurves = np.array([Tf*pdat['T'] + Tf*Tf*pdat['T2'] for i,pdat in plotdat.iterrows()])
        Pcurves = np.array([Pf*pdat['P'] + Pf*Pf*pdat['P2'] for i,pdat in plotdat.iterrows()])

        Tcurvep = np.percentile(np.diff(Tcurves,1),50,axis=0)
        Pcurvep = np.percentile(np.diff(Pcurves,1),50,axis=0)
        Tcurvep_low = np.percentile(np.diff(Tcurves,1),5,axis=0)
        Pcurvep_low = np.percentile(np.diff(Pcurves,1),5,axis=0)
        Tcurvep_high = np.percentile(np.diff(Tcurves,1),95,axis=0)
        Pcurvep_high = np.percentile(np.diff(Pcurves,1),95,axis=0)

        if subtractTrue:
            Tcurvep = Tcurvep-Tcurvetrued
            Pcurvep = Pcurvep-Pcurvetrued
            Tcurvep_low = Tcurvep_low-Tcurvetrued
            Tcurvep_low = Tcurvep_low-Tcurvetrued
            Pcurvep_high = Pcurvep_high-Pcurvetrued
            Pcurvep_high = Pcurvep_high-Pcurvetrued

        plotdat['Tmarginal_02']      = plotdat['T']      + 2*plotdat['T2']     *0.2
        plotdat['Tmarginal_02_true'] = plotdat['T_true'] + 2*plotdat['T2_true']*0.2

        plotdat['Tmarginal_08']      = plotdat['T']      + 2*plotdat['T2']     *0.8
        plotdat['Tmarginal_08_true'] = plotdat['T_true'] + 2*plotdat['T2_true']*0.8

        plotdat['Pmarginal_02']      = plotdat['P']      + 2*plotdat['P2']     *0.2
        plotdat['Pmarginal_02_true'] = plotdat['P_true'] + 2*plotdat['P2_true']*0.2

        plotdat['Pmarginal_08']      = plotdat['P']      + 2*plotdat['P2']     *0.8
        plotdat['Pmarginal_08_true'] = plotdat['P_true'] + 2*plotdat['P2_true']*0.8

        l02 = ' (cold)'
        l08 = ' (warm)'

        sns.kdeplot(plotdat['Tmarginal_02']/plotdat['Tmarginal_02_true'],ax=ax3,color='r',
                    alpha=0.4,lw=2,linestyle=info['line'],label=info['name']+l02)
        sns.kdeplot(plotdat['Tmarginal_08']/plotdat['Tmarginal_08_true'],ax=ax3,color='r',
                    alpha=1,lw=2,linestyle=info['line'],label=info['name']+l08)
        ax3.set_xlabel('Fraction of true marginal T effect',fontsize=13)
        ax3.set_ylabel('Density',fontsize=13)
        ax3.set_xlim([-0.5,2])
        ax3.plot([1,1],ax3.get_ylim(),'--k')

        sns.kdeplot(plotdat['Pmarginal_02']/plotdat['Pmarginal_02_true'],ax=ax4,color='b',
                    alpha=0.3,lw=2,linestyle=info['line'],label=info['name']+l02)
        sns.kdeplot(plotdat['Pmarginal_08']/plotdat['Pmarginal_08_true'],ax=ax4,color='b',
                    alpha=1,lw=2,linestyle=info['line'],label=info['name']+l08)
        ax4.set_xlabel('Fraction of true marginal P effect',fontsize=13)
        ax4.set_ylabel('Density',fontsize=13)
        ax4.set_xlim([-0.5,2])
        ax4.plot([1,1],ax4.get_ylim(),'--k')

ax3.annotate('({})'.format('c'),(0.02,0.05),
                    xycoords='axes fraction',fontsize=14)
ax4.annotate('({})'.format('d'),(0.02,0.05),
                    xycoords='axes fraction',fontsize=14)

plt.sca(ax3)
plt.legend(loc=2,fontsize=10)

plt.subplots_adjust(wspace=0,hspace=0.1,top=0.95,bottom=0.1,left=0.02,right=0.92)
plt.savefig('../figures/figure2.png',dpi=600)


