"""
This script runs the linear idealized model and makes figure 4
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
from matplotlib import cm
import time
import pdb
import quadoutcome as qo
import xarray as xr
import cartopy.crs as ccrs
import seaborn as sns

def modelrun(Nt,Nx,use_FE,cov,Pcoef,Tcoef,Pstderr,Tstderr,outcome_const,enso_demo):
    """
    modelrun(Nt,Nx,use_FE,cov,Pcoef,Tcoef,Pstderr,Tstderr,outcome_const):
    """
    #   Initialize
    data = pd.DataFrame(columns=['P','T','Perr','Terr','site','outcome'])

    #   Create data
    for i in range(Nx):
        T, P = np.random.multivariate_normal(mean=[10*np.random.randn(),10*np.random.randn()],
                                                   cov=[[1,cov],[cov,1]],size=Nt).T
        
        fe1 = i*np.ones(Nt)
        fe2 = np.arange(Nt)
    
        #   Add P error if selected above
        if 'std' in Pstderr:
            err = float(Pstderr.split('_')[0])
            Perr = P + err*np.std(P)*np.random.randn(Nt)
        else:
            P = P*1.

        #   Add P error if selected above
        if 'std' in Tstderr:
            err = float(Tstderr.split('_')[0])
            Terr = T + err*np.std(T)*np.random.randn(Nt)
        else:
            P = P*1.
    
        oc = outcome_const*np.random.randn()
        if enso_demo:
            midi = int(Nt/2)
            T[midi] += 3*np.std(T)*np.sign(cov)

        outcome = Tcoef*T + Pcoef*P + oc

        data = data.append(pd.DataFrame(np.array([P,T,Perr,Terr,outcome,fe1,fe2]).T,
                                        columns=['P','T','Perr','Terr','outcome','site','time']))
    
    data = data.reset_index()
    
    #   Demean (fixed effects)
    yx = data.copy()
    #   Site fixed effects
    if use_FE==1:
        demeaner1 = yx.groupby('site').mean()
        demeaner1 = yx[['site']].join(demeaner1,on='site').drop(['site'],axis=1)
        yxd = yx - demeaner1

    #   Site + Time fixed effects
    elif use_FE==2:
        demeaner1 = yx.drop('time', axis=1).groupby('site').mean()
        demeaner1 = yx[['site']].join(demeaner1, on='site').drop(['site'], axis=1)
        demeaner2 = yx.drop('site', axis=1).groupby('time').mean()
        demeaner2 = yx[['time']].join(demeaner2, on='time').drop(['time'], axis=1)
        yxd = yx - demeaner1 - demeaner2 + yx.mean()

    else:
        yxd = yx
    
    #   Run model
    #   formula = 'outcome~Terr+Perr-1'
    formula = 'outcome~Terr+Perr-1'
    mod = smf.ols(formula,data=yxd)
    res = mod.fit(cov_type='cluster',cov_kwds={'groups':data['site']})
    
    #   pdb.set_trace()
    return res, yx, yxd


#   #   Options
Nt = 50
Nx = 30
Nreps = 10
use_FE = 1  # 1 means site FE, 2 means site and time FE
#   covs = np.linspace(-0.9,0.9,19)
covs = np.linspace(-0.9,0.9,11)
#   Pcoefs = np.linspace(0,1,6)
#   Tcoefs = np.linspace(1,0,6)
Pcoefs = np.linspace(0,1,3)
Tcoefs = np.linspace(1,0,3)
#   Pstdfrac = np.linspace(0,3,13)
Pstdfrac = np.linspace(0,1,11)
Pstderrs = [str(i)+'_std' for i in Pstdfrac] #   Options: 'none','#_std'
Tstdfrac = 0
Tstderr = str(Tstdfrac)+'_std'
outcome_const = 20
enso_demo=False

Nsim = Nreps*len(covs)*len(Pcoefs)*len(Pstderrs)

savelist = []
ct=0
start = time.time()
for rep in range(Nreps):
    for cov in covs:
        for Pstderr in Pstderrs:
            for (Tcoef,Pcoef) in zip(Tcoefs,Pcoefs):
                ct+=1
                res, yx, yxd = modelrun(Nt=Nt,Nx=Nx,use_FE=use_FE,cov=cov,Pcoef=Pcoef,Tcoef=Tcoef,
                                        Pstderr=Pstderr,Tstderr=Tstderr,outcome_const=outcome_const,
                                        enso_demo=enso_demo)
                output = {'cov':cov,'Pstderr':Pstderr,'Tstderr':Tstderr,'Pcoef':Pcoef,
                          'Tcoef':Tcoef,'res':res,'rep':rep}
    
                savelist.append(output)
    
                if ct % 10 == 0:
                    end = time.time()
                    tdif = end-start
                    start = time.time()
                    print('Finished {} simulations, last 10 in {:.2f} sec. \
                           Remaining: ~{:.1f} min'.format(ct,tdif,(Nsim-ct)/10*tdif/60))


#   Format output into plottable format
Tcoefdif = np.nan*np.ones((len(covs),len(Pstderrs),Nreps))
Pcoefdif = np.nan*np.ones((len(covs),len(Pstderrs),Nreps))
Puse = 0.5
Tuse = 0.5

for out in savelist:
    if (np.abs(out['Tcoef']-Tuse) < 1e-6) & \
       (np.abs(out['Pcoef']-Puse) < 1e-6):

        covi  = np.where(np.abs(covs-out['cov'])<1e-6)[0][0]
        Pstdi = [i for i,er in enumerate(Pstderrs) if er==out['Pstderr']]

        Tcoefdif[covi,Pstdi,out['rep']] = out['res'].params['Terr']
        Pcoefdif[covi,Pstdi,out['rep']] = out['res'].params['Perr']

#   Plot contours
plt.figure(figsize=(10,7))

ax=plt.subplot(223)
CF = ax.contourf(x,covs,Tcoefdif.mean(-1)/Tuse,50,cmap='bwr',vmin=1-1,vmax=1+1)
cticks = np.arange(0.2,1.8,0.2)
cb1 = plt.colorbar(CF,ticks=cticks)
cb1.set_label(r'$\beta _T$ / $\beta _{Ttrue}$',rotation=270,labelpad=20,fontsize=13)
conticks = [0.5,0.8,0.9,1.1,1.2,1.5]
CS = ax.contour(x, covs, Tcoefdif.mean(-1)/Tuse, 6, colors='k',levels=conticks)
ax.clabel(CS, fontsize=12, inline=True,fmt='%1.2f')
plt.xlabel(xlab,fontsize=13)
plt.ylabel('cor(T,P)',fontsize=14)
plt.annotate('(c)',(0.02,0.92),xycoords='axes fraction',fontsize=14)

ax2=plt.subplot(224)
let2 = 'd'
CF = ax2.contourf(x,covs,Pcoefdif.mean(-1)/Puse,50,cmap='bwr',vmin=1-1.2,vmax=1+1.2)
cb2 = plt.colorbar(CF,ticks=cticks)
cb2.set_label(r'$\beta _P$ / $\beta _{Ptrue}$',rotation=270,labelpad=20,fontsize=13)
CS = ax2.contour(x, covs, Pcoefdif.mean(-1)/Puse, 6, colors='k')
ax2.clabel(CS, fontsize=12, inline=True,fmt='%1.1f')
plt.xlabel(xlab,fontsize=13)
plt.ylabel('cor(T,P)',fontsize=13)
plt.annotate('({})'.format(let2),(0.02,0.92),xycoords='axes fraction',fontsize=14)

plt.tight_layout()

#   Map
udelcor = xr.open_dataarray('../data/correlations/UDEL_1960_1989_cor.nc')
ax3 = plt.subplot(2,2,1,projection=ccrs.Robinson(),frameon=False)
tls = [0.02,0.5]    #      Top left coords
tb = 0.02
tw = np.diff(tls)[0]-0.01+0.3
th = 0.5
corlim = 0.75
cfrac = 0.027
im = udelcor.plot(vmin=-corlim,vmax=corlim,cmap='PuOr_r',transform=ccrs.PlateCarree(),
                  add_colorbar=False)
             #   cbar_kwargs={'fraction':cfrac,'pad':0.08,'label':'cor(T,P)'})
cb3 = plt.colorbar(im, fraction=cfrac, pad=0, extend='both')
cb3.set_label(label='UDEL cor(T,P)', size=13,rotation=270,va='bottom')
cb3pos0 = cb3.ax.get_position().extents
cb3.ax.set_position(Bbox.from_extents([0.418,0.63]+list(cb3pos0)[2:]))
ax3.coastlines(lw=0.5)
ax3.set_position(Bbox.from_extents([0.03,0.55,0.4,1]))
ax3.annotate('({})'.format('a'),(0.08,1.22),xycoords='axes fraction',fontsize=14)

#   Histogram
wadjb = 0.05
histcor = xr.open_dataarray('../data/correlations/CESM_1960_1989_cor.nc')
ssp2cor = xr.open_dataarray('../data/correlations/CESM_2040_2069_cor.nc')
pop = xr.open_dataset('../data/gdp/combxr_05deg.nc')['pop'].sel({'time':2015}).where(ssp2cor)
ax4 = plt.subplot(2,2,2)
cols = {'UDEL historical':'#4689F5','CESM historical':'#F0A771','CESM transient':'#9167EF'}
for k,d in {'UDEL historical':udelcor,'CESM historical':histcor,'CESM transient':ssp2cor}.items():
    vals = np.ndarray.flatten(d.values)
    wts  = np.ndarray.flatten(pop.values)
    nni = ~np.isnan(vals*wts)
    sns.kdeplot(x=vals[nni],weights=wts[nni],ax=ax4,label=k,color=cols[k],clip_on=False)
plt.plot([0,0],ax4.get_ylim(),'--k',alpha=0.6)
plt.legend(loc=1,fontsize=9,frameon=False)
ax4.set_position([0.5844,0.63,0.304,0.35])
ax4.annotate('({})'.format('b'),(0.02,0.90),xycoords='axes fraction',fontsize=14)
plt.ylabel('Density',fontsize=13)
plt.xlabel('cor(T,P) of pop-weighted grid cells',fontsize=13)


plt.savefig('../figures/figure4.png',dpi=600)

