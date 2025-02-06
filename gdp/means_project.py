"""
This file creates Figure 8.
It is quite long and the plotting routine takes a few minutes
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import xarray as xr
from linearmodels import PanelOLS
import pdb
import cartopy.crs as ccrs
import statsmodels.formula.api as smf
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

ctrytime = True  #   Country*Time
specs    = ['T', 'T + T*Tmean', 'Tanom + Tanom*Tmean', 'T + Tanom2']
mean0s   = {'T':True, 'T + T*Tmean':True, 'Tanom + Tanom*Tmean':True, 'T + Tanom2':True}
plotvar = 'gdpg' #   Do not change

#   Read data growth
gdp  = pd.read_csv('../data/gdp_fits/get_fe_ds_country_id_anom.csv',index_col=0)
gdpg = pd.read_csv('../data/gdp_fits/get_fe_ds_country_id_growth_anom.csv',index_col=0)

gdp  = gdp.rename(columns={'gdppcg':'loggdppc'}) #    Fix label
gdp  = gdp.rename(columns={'T':'Tanom','P':'Panom','Tabs':'T','Pabs':'P',
                           'T2':'Tanom2','P2':'Panom2','Tabs2':'T2','Pabs2':'P2'})
gdpg = gdpg.rename(columns={'T':'Tanom','P':'Panom','Tabs':'T','Pabs':'P',
                            'T2':'Tanom2','P2':'Panom2','Tabs2':'T2','Pabs2':'P2'})

gdp['country_id']  = gdp['country_id'].astype(str)
gdpg['country_id'] = gdpg['country_id'].astype(str)

gdp  = gdp.set_index(['country_id','time'],drop=False)
gdpg = gdpg.set_index(['country_id','time'],drop=False)

ssp = pd.read_csv('../data/gdp_fits/SspDb_country_data_2013-06-12.csv')

def regres(formula,dat,weight=None):
    dat_panel   = PanelOLS.from_formula(formula,dat,weights=weight).fit(
                        cov_type="clustered", cluster_entity=True)
    coef_dat    = dat_panel.params
    return coef_dat, dat_panel

coefs    = {}
panels   = {}
formulas = {}
for spec in specs:
    coefs[spec]    = {}
    panels[spec]   = {}
    formulas[spec] = {}

#   Run standard regressions and anomaly/interaction regressions
    if ctrytime:
        effects = ' + time:country_id + EntityEffects'
    else:
        effects = ' + EntityEffects + TimeEffects'
    
    if spec=='T':
        formula_main = 'T + T2 + P + P2'
    elif spec.replace(' ','')=='T+T*Tmean':
        formula_main = 'T + T2 + T:Tmean + T2:Tmean + ' +\
                       'P + P2 + P:Pmean + P2:Pmean'
    elif spec.replace(' ','')=='Tanom+Tanom*Tmean':
        formula_main = 'Tanom + Tanom2 + Tanom:Tmean + Tanom2:Tmean + ' +\
                       'Panom + Panom2 + Panom:Pmean + Panom2:Pmean'
    elif spec.replace(' ','')=='T+Tanom2':
        formula_main = 'T + T2 + Tanom2 + ' +\
                       'P + P2 + Panom2'
    
    formula_gdp  = 'loggdppc ~ ' + formula_main + effects
    formula_gdpg = 'gdppcg   ~ ' + formula_main + effects
    
    coef_gdp,   panel_gdp   = regres(formula_gdp,gdp)
    coef_gdpg,  panel_gdpg  = regres(formula_gdpg,gdpg)

    coefs[spec]['gdp']   = coef_gdp
    coefs[spec]['gdpg']  = coef_gdpg

    panels[spec]['gdp']  = panel_gdp
    panels[spec]['gdpg'] = panel_gdpg

    formulas[spec]['gdp']  = formula_gdp
    formulas[spec]['gdpg'] = formula_gdpg


#   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   

def eval_spec(coef,inputs={},mean0=False):
    terms = []
    for inp in inputs.keys():
        usei = [inp in x for x in coef.index.str.split(':')]
        terms+=list(coef.keys()[usei])

    terms = np.unique(terms)
    out = 0
    termvals = {}
    for term in terms:
        multterm = 1
        for subterm in term.split(':'):
            multterm *= inputs[subterm]
        termvals[term] = coef.loc[term]*multterm
        out += termvals[term]

    if mean0:
        out-=np.mean(out)

    return out, termvals

def proj_growth(coef,inputs,sspgdp,baseyrs,projyrs,subtract=False,mean0=False,excludeterms=[]):
    """
    Project GDP from growth
    excludeterms are replaced by base values instead of projected values in the projections
    """

    if subtract:
        sign = -1
    else:
        sign = 1

    includes_Tmean = any([True for c in coef.index.values if 'mean' in c])

    out = sspgdp['gdppc'].copy()
    for yi,y in enumerate(range(projyrs[0]+1,projyrs[1]+1)):

        #   Allow possibility for base year to move
        if type(baseyrs[0])==list:
            try:
                baseyrsu = baseyrs[yi]
            except:
                pdb.set_trace()
        else:
            baseyrsu = baseyrs.copy()

        #   Select inputs in loop in case baseline is moving
        inputs1={}
        for k,v in inputs.items():
            if 'year' in v.coords:
                inputs1[k] = v.sel(year=range(baseyrsu[0],projyrs[1]+1))
            else:
                inputs1[k] = v

        #   Move projection into loop for moving mean
        if includes_Tmean:
            if 'year' in inputs['Tmean'].coords:
                inputs1['Tmean'] = inputs['Tmean'].sel(year=y).drop('year')
        proj0, proj_terms = eval_spec(coef,inputs=inputs1,mean0=mean0)
        base              = proj0.sel(year=range(baseyrsu[0],baseyrsu[1]+1)).mean('year')
        base_terms        = {term:proj_term.sel(year=range(baseyrsu[0],baseyrsu[1]+1)).mean('year') 
                                 for term,proj_term in proj_terms.items()}

        proj = 0
        proj_orig = 0
        for term, proj_term in proj_terms.items():
            proj_orig += proj_term
            if term in excludeterms:
                proj += (proj_term/proj_term) * base_terms[term]
            else:
                proj += proj_term
        if mean0:
            proj-=np.mean(proj_orig)

        d = proj - base

        oldGDP   = out.sel(year=y-1).drop('year')
        change   = (1 + sspgdp['gdppcg'].sel(year=y).drop('year')+sign*d.sel(year=y).drop('year'))
        change   = change.sel(country_id=sspgdp.country_id)

        out.loc[{'year':y}]   = oldGDP   * change
    out.name = 'gdppc_clim'
    return out


#   Convert to xarray, run projections
#   GDP means using gdp, not gdppc because anoms calculated on full dataset in other script
climproj = xr.open_dataset('../data/gdp/projections/combxrag_country.nc')
#   Due to slight differences in population, set climproj overlap with gdpg to be gdpg values
gdpgmerge = gdpg[['gdppc','pop','T','P']].to_xarray().rename({'gdppc':'GDPpc','time':'year'})
gdpgmerge = gdpgmerge.assign_coords({'country_id':gdpgmerge.country_id.astype(float)})
climproj[['GDPpc','pop','T','P']].loc[{'year':gdpgmerge.year.values,'country_id':gdpgmerge.country_id.values}] = gdpgmerge

#   Rename countries to usable names
rastermeta = pd.read_csv('../data/geo/raster_id_meta_05deg.csv')

gdpcountrymap = rastermeta[['GID_0','country_id']].drop_duplicates().set_index(
                    'country_id').to_dict()['GID_0']
climcountry = [gdpcountrymap[c] for c in climproj.country_id.values]
climproj = climproj.assign_coords({'country_id':climcountry})


gdpmeans = gdp[['Tmean','Pmean']].to_xarray().isel(time=-1).drop('time')
gdpmeans = gdpmeans.assign_coords({'country_id':gdpmeans.country_id.astype(float)})
gdpcountry = [gdpcountrymap[c] for c in gdpmeans.country_id.values.astype(float)]
gdpmeans = gdpmeans.assign_coords({'country_id':gdpcountry})

ctryshp = gpd.read_file('../data/gadm/gadm36_0_lowlowres.shp')
ctryshp = ctryshp.rename(columns={'NAME_0':'country','GID_0':'country_id'})

climproj['Tanom_gdp'] = climproj['T'] - gdpmeans['Tmean']
climproj['Panom_gdp'] = climproj['P'] - gdpmeans['Pmean']
climproj['Tmean_gdp'] = gdpmeans['Tmean']
climproj['Pmean_gdp'] = gdpmeans['Pmean']

climproj['Tmean_roll']  = climproj['T'].rolling(year=25).mean() 
climproj['Pmean_roll']  = climproj['P'].rolling(year=25).mean() 
climproj['Tanom_roll']  = climproj['T'] - climproj['Tmean_roll']
climproj['Panom_roll']  = climproj['P'] - climproj['Pmean_roll']
climproj['Tanom_roll2'] = climproj['Tanom_roll']**2
climproj['Panom_roll2'] = climproj['Panom_roll']**2


#   Add all the square terms for simplicity
climproj['T2'] = climproj['T']**2
climproj['P2'] = climproj['P']**2
climproj['Tanom_gdp2'] = climproj['Tanom_gdp']**2
climproj['Panom_gdp2'] = climproj['Panom_gdp']**2

sspgdp = []
for v in ['GDP|PPP','Population']:
    ssp_v = ssp[(ssp.VARIABLE==v) &
                  (ssp.SCENARIO.str.startswith('SSP2')) & 
                  (ssp.MODEL=='OECD Env-Growth')]
    ssp_v = pd.melt(ssp_v,id_vars=['REGION','MODEL','SCENARIO','VARIABLE','UNIT'],
                   var_name='year',value_name=v)
    ssp_v = ssp_v.drop(columns=['VARIABLE','UNIT','MODEL','SCENARIO']).set_index(
                   ['REGION','year'])
    ssp_v = ssp_v.to_xarray().rename({'REGION':'country_id'})
    ssp_v = ssp_v.assign_coords(year=ssp_v.year.astype(int))
    ssp_v = ssp_v.sel(year=ssp_v['year'].values>=2010)
    ssp_v = ssp_v.interp(year=range(2010,2101))
    sspgdp.append(ssp_v)
sspgdp = xr.merge(sspgdp)
sspgdp['GDP|PPP']    *= 1e9
sspgdp['Population'] *= 1e6
sspgdp['gdppc']  = sspgdp['GDP|PPP']/sspgdp['Population']
sspgdp['gdppcg'] = (sspgdp['gdppc'] / sspgdp['gdppc'].shift(year=1)) - 1
overlap = list(set(sspgdp.country_id.values).intersection(climproj.country_id.values))
sspgdp = sspgdp.sel(country_id=overlap)

aveap = '_gdp'
    
proj2050s          = {}
proj2050svar       = {}
proj2050smean      = {}
proj2050s_roll     = {}
proj2050s_rollvar  = {}
proj2050s_rollmean = {}
projs              = {}
projsvar           = {}
projsmean          = {}
projs_roll         = {}
projs_rollvar      = {}
projs_rollmean     = {}
hindcasts          = {}
hindcasts_full     = {}
for spec in specs:

    baseyrs = [2005,2015]
    projyrs = [2015,2090]
    if spec.replace(' ','')=='T':
        inputs     = {'T':climproj['T'],'T2':climproj['T2'],
                      'P':climproj['P'],'P2':climproj['P2']}
        varexclude  = []
        meanexclude = ['T','T2','P','P2']
    elif spec.replace(' ','')=='T+T*Tmean':
        inputs     = {'T':climproj['T'],'T2':climproj['T2'],
                      'P':climproj['P'],'P2':climproj['P2'],
                      'Tmean':climproj['Tmean'+aveap],'Pmean':climproj['Pmean'+aveap]
                     }
        varexclude  = ['T:Tmean','T2:Tmean','P:Pmean','P2:Pmean']
        meanexclude = ['T','T2','P','P2']
    elif spec.replace(' ','')=='Tanom+Tanom*Tmean':
        inputs     = {'Tanom':climproj['Tanom'+aveap],'Panom':climproj['Panom'+aveap],
                      'Tanom2':climproj['Tanom'+aveap+'2'],'Panom2':climproj['Panom'+aveap+'2'],
                      'Tmean':climproj['Tmean'+aveap],'Pmean':climproj['Pmean'+aveap]
                      }
        varexclude  = ['Tanom:Tmean','Tanom2:Tmean','Panom:Pmean','Panom2:Pmean']
        meanexclude = ['Tanom','Tanom2','Panom','Panom2']
    elif spec.replace(' ','')=='T+Tanom2':
        inputs     = {'T':climproj['T'],'T2':climproj['T2'],
                      'P':climproj['P'],'P2':climproj['P2'],
                      'Tanom2':climproj['Tanom'+aveap+'2'],'Panom2':climproj['Panom'+aveap+'2'],
                      }
        varexclude  = ['Tanom2','Panom2']
        meanexclude = ['T','T2','P','P2']

    climproj = climproj.sel(country_id=overlap)
    proj     = proj_growth(coefs[spec][plotvar],inputs,sspgdp,
                           baseyrs=baseyrs,projyrs=projyrs,
                           subtract=False,mean0=mean0s[spec],excludeterms=[])
    projvar  = proj_growth(coefs[spec][plotvar],inputs,sspgdp,
                           baseyrs=baseyrs,projyrs=projyrs,
                           subtract=False,mean0=mean0s[spec],excludeterms=varexclude)
    projmean = proj_growth(coefs[spec][plotvar],inputs,sspgdp,
                           baseyrs=baseyrs,projyrs=projyrs,
                           subtract=False,mean0=mean0s[spec],excludeterms=meanexclude)
    proj2050 = (proj.sel(year=range(2045,2055)) / sspgdp.sel(year=range(2045,2055)))['gdppc']
    proj2050 = proj2050.mean('year')

    proj2050var = (projvar.sel(year=range(2045,2055)) / sspgdp.sel(year=range(2045,2055)))['gdppc']
    proj2050var = proj2050var.mean('year')

    proj2050mean = (projmean.sel(year=range(2045,2055)) / sspgdp.sel(year=range(2045,2055)))['gdppc']
    proj2050mean = proj2050mean.mean('year')

    projs[spec]     = proj
    projsvar[spec]  = projvar
    projsmean[spec] = projmean

    proj2050s[spec]     = proj2050
    proj2050svar[spec]  = proj2050var
    proj2050smean[spec] = proj2050mean

    #   Repeat using rolling means where means are used
    if spec.replace(' ','')=='T':
        inputs     = {'T':climproj['T'],'T2':climproj['T2'],
                      'P':climproj['P'],'P2':climproj['P2']}
        varexclude  = []
        meanexclude = ['T','T2','P','P2']
        baseyrs = [[py-25,py] for py in range(projyrs[0],projyrs[1]+1)]
    elif spec.replace(' ','')=='T+T*Tmean':
        inputs     = {'T':climproj['T'],'T2':climproj['T2'],
                      'P':climproj['P'],'P2':climproj['P2'],
                      'Tmean':climproj['Tmean_roll'],'Pmean':climproj['Pmean_roll']
                     }
        varexclude  = ['T:Tmean','T2:Tmean','P:Pmean','P2:Pmean']
        meanexclude = ['T','T2','P','P2']
    elif spec.replace(' ','')=='Tanom+Tanom*Tmean':
        inputs     = {'Tanom':climproj['Tanom_roll'],'Panom':climproj['Panom_roll'],
                      'Tanom2':climproj['Tanom_roll2'],'Panom2':climproj['Panom_roll2'],
                      'Tmean':climproj['Tmean_roll'],'Pmean':climproj['Pmean_roll']
                      }
        varexclude  = ['Tanom:Tmean','Tanom2:Tmean','Panom:Pmean','Panom2:Pmean']
        meanexclude = ['Tanom','Tanom2','Panom','Panom2']
    elif spec.replace(' ','')=='T+Tanom2':
        inputs     = {'T':climproj['T'],'T2':climproj['T2'],
                      'P':climproj['P'],'P2':climproj['P2'],
                      'Tanom2':climproj['Tanom_roll2'],'Panom2':climproj['Panom_roll2'],
                      }
        varexclude  = ['Tanom2','Panom2']
        meanexclude = ['T','T2','P','P2']

    climproj = climproj.sel(country_id=overlap)
    proj     = proj_growth(coefs[spec][plotvar],inputs,sspgdp,
                           baseyrs=baseyrs,projyrs=projyrs,
                           subtract=False,mean0=mean0s[spec],excludeterms=[])
    projvar  = proj_growth(coefs[spec][plotvar],inputs,sspgdp,
                           baseyrs=baseyrs,projyrs=projyrs,
                           subtract=False,mean0=mean0s[spec],excludeterms=varexclude)
    projmean = proj_growth(coefs[spec][plotvar],inputs,sspgdp,
                           baseyrs=baseyrs,projyrs=projyrs,
                           subtract=False,mean0=mean0s[spec],excludeterms=meanexclude)
    proj2050     = proj.sel(year=range(2045,2055)) / sspgdp.sel(year=range(2045,2055))['gdppc']
    proj2050     = proj2050.mean('year')
    proj2050var  = projvar.sel(year=range(2045,2055)) / sspgdp.sel(year=range(2045,2055))['gdppc']
    proj2050var  = proj2050var.mean('year')
    proj2050mean = projmean.sel(year=range(2045,2055)) / sspgdp.sel(year=range(2045,2055))['gdppc']
    proj2050mean = proj2050mean.mean('year')

    projs_roll[spec]     = proj
    projs_rollvar[spec]  = projvar
    projs_rollmean[spec] = projmean
    proj2050s_roll[spec]     = proj2050
    proj2050s_rollvar[spec]  = proj2050var
    proj2050s_rollmean[spec] = proj2050mean


    #   Hindcasts
    if plotvar=='gdp':
        dat_fixclim = gdp.copy()
    elif plotvar=='gdpg':
        dat_fixclim = gdpg.copy()

    vs = formulas[spec][plotvar].split('~')[1].split('+')[:-2]
    vsu = []
    for v in vs:
        vsu += v.split(':')
    vs = np.unique([v.strip() for v in vsu if not v.strip().endswith('2')])
    for v in vs:
        vcolname = 'fix_'+v
        meanv = dat_fixclim.pivot(index='time',columns='country_id',values=v).loc[
                        #   1995:2005].mean()
                        2005:2015].mean()
        meanv.name = vcolname
        dat_fixclim    = dat_fixclim.drop(columns='country_id').merge(meanv,on='country_id')
        dat_fixclim    = dat_fixclim.reset_index().set_index(['country_id','time'],drop=False)
        dat_fixclim[v] = dat_fixclim[vcolname]
        dat_fixclim[v+'2'] = dat_fixclim[v]**2
    if plotvar=='gdp':
        predict         = panels[spec][plotvar].predict(data=gdp)['predictions']
    if plotvar=='gdpg':
        predict         = panels[spec][plotvar].predict(data=gdpg)['predictions']
    predict_fixclim = panels[spec][plotvar].predict(data=dat_fixclim)['predictions']
    hindcast = (predict - predict_fixclim).swaplevel().loc[range(2005,2015)] * 100
    hindcasts_full[spec] = hindcast 
    
    if plotvar=='gdpg':
        hindcast = hindcast.reset_index().pivot(
                        index='time',columns='country_id',values='predictions')
        hindcast = (1+hindcast).cumprod().mean() - 1
    else:
        hindcast = hindcast.groupby('country_id').mean()
    hindcasts[spec] = hindcast


#   Fix countries and merge with shapefiles to allow for projection
ctryshp = gpd.read_file('../data/gadm/gadm36_0_lowlowres.shp')
ctryshp = ctryshp.to_crs(ccrs.Robinson())

#   Add country identifier
gdpg0 = gdpg.copy()
shpmeta = pd.read_csv('../data/geo/raster_id_meta_05deg.csv')
gdp.index = range(len(gdp))
gdp['country_id'] = gdp['country_id'].astype(float).astype(int)
gdp = gdp.merge(shpmeta[['GID_0','country_id']].drop_duplicates(),on='country_id')
gdpg.index = range(len(gdpg))
gdpg['country_id'] = gdpg['country_id'].astype(float).astype(int)
gdpg = gdpg.merge(shpmeta[['GID_0','country_id']].drop_duplicates(),on='country_id')

shpmetau = shpmeta[['GID_0','country_id']].drop_duplicates()

for spec in specs:
    hindcasts[spec] = hindcasts[spec].reset_index().rename(columns={0:'hindcast'})
    hindcasts[spec]['country_id'] = hindcasts[spec]['country_id'].astype(float).astype(int)
    hindcasts[spec] = hindcasts[spec].merge(shpmetau[['GID_0','country_id']],on='country_id')
    hindcasts[spec]['country_id'] = hindcasts[spec]['GID_0']
    hindcasts[spec] = ctryshp.merge(hindcasts[spec],on='GID_0')

#   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   
#   Create latex table of panels
Nparams = [4,8,8,6] 
latout = []
latindex = ['T','','T2','','P','','P2','','T:Tmean','','T2:Tmean','','P:Pmean','','P2:Pmean','',
            'Tanom','','Tanom2','','Panom','','Panom2','','Tanom:Tmean','','Tanom2:Tmean','',
            'Panom:Pmean','','Panom2:Pmean','']
latindex = np.array(latindex)
latout = pd.DataFrame(index=latindex,columns=range(len(Nparams)))
for pi, panel in enumerate(panels):
    m = pd.DataFrame(panels[specs[pi]]['gdpg'].summary.tables[1].data).iloc[:Nparams[pi]+1]
    m.columns = m.iloc[0].values
    m = m.iloc[1:]
    m = m.set_index('')

    for p,row in m.iterrows():
        curi = np.where(latindex==p)[0][0]
        latout.iloc[curi,pi] = row['Parameter']
        latout.iloc[curi+1,pi] = '({})'.format(row['Std. Err.'])

latout.style.to_latex()


#   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   
    

#   Plot both curves for each dataset
Trange1 = np.linspace(-5,30)
Trange  = np.arange(-5,31,5)
Tanom   = np.linspace(-3,3)

Prange1 = np.linspace(0,300)
Prange  = np.arange(20,200,20)
Panom   = np.linspace(-10,10,20)

map_name = {'gdp':'GDPpc levels','gdpg':'GDPpc growth'}
#   plt.figure(figsize=(3*len(specs),9))
plt.figure(figsize=(3.5*len(specs)+1,9))
trackax = []
Tvars  = {'T':Trange1,'Tanom':Tanom,'T2':Trange1**2,'Tanom2':Tanom**2}
sct = 0
alpha = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v']
alpha = ['({})'.format(a) for a in alpha]
titles = {
    'T':'$T_{ct} + T_{ct}^2$',
    'T + T*Tmean':'$T_{ct} + T_{ct}^2 + \overline{T}_{c(t)}(T_{ct} + T_{ct}^2)$',
    'Tanom + Tanom*Tmean':'$T_{a,ct}+T_{a,ct}^2+\overline{T}_{c(t)}(T_{a,ct}+T_{a,ct}^2)$',
    'T + Tanom2':'$T_{ct} + T_{ct}^2 + T_{a,ct}^2$',
    }
#   Tvars  = {'P':Prange1,'Panom':Panom,'P2':Prange1**2,'Panom2':Panom**2}
norm = Normalize(vmin=0,vmax=30)
cmap = plt.get_cmap('coolwarm')
sm = ScalarMappable(cmap=cmap,norm=norm)
ylim = (-10,7)

for si,spec in enumerate(specs):
    sct+=1
    Tvarsu = Tvars.copy()
    curax = plt.subplot(5,len(specs),si+1)
    trackax.append(curax)
    x = Tanom
    for t in Trange:
        if spec=='T':
            Tvarsu['T']      = t + Tanom
            Tvarsu['T2']     = Tvarsu['T']**2

        elif spec.replace(' ','')=='T+T*Tmean':
            Tvarsu['Tmean']  = t
            Tvarsu['T']      = t + Tanom
            Tvarsu['T2']     = Tvarsu['T']**2

        elif spec.replace(' ','')=='Tanom+Tanom*Tmean':
            Tvarsu['Tmean']  = t
            Tvarsu['Tanom']  = Tanom
            Tvarsu['Tanom2'] = Tanom**2

        elif spec.replace(' ','')=='T+Tanom2':
            Tvarsu['Tmean']  = t
            Tvarsu['T']      = t+Tanom
            Tvarsu['T2']     = Tvarsu['T']**2


        curve,terms = eval_spec(coefs[spec][plotvar],Tvarsu,mean0s[spec])
        plt.plot(x,curve*100,color=sm.to_rgba(t))
        plt.gca().set_ylim(ylim)
        plt.plot([0,0],ylim,'--k',alpha=0.4)

    cb=plt.colorbar(sm,ax=curax)
    plt.xlabel('$\Delta$ T (C)')
    if spec=='T':
        plt.ylabel('$\Delta GDP_{pc}$ - $\overline{\Delta GDP_{pc}}$ [%]')
        cb.set_label('$T$',rotation=270,ha='center')
    elif spec.replace(' ','')=='T+Tanom2':
        cb.set_label('$T$',rotation=270,ha='center')
    else:
        cb.set_label('$\overline{T}$',rotation=270,ha='center')
    #   plt.title(formulas[spec][plotvar])
    plt.title(titles[spec])
    plt.annotate(alpha[sct-1],(0.01,0.95),xycoords='axes fraction',ha='left',va='top')


#   Plot hindcasts
for si,spec in enumerate(specs):
    sct+=1
    curax = plt.subplot(5,len(specs),si+1+len(specs),projection=ccrs.Robinson())
    trackax.append(curax)
    (hindcasts[spec]).plot('hindcast',ax=curax,vmin=-3,vmax=3,cmap='coolwarm_r',
                        legend=True,legend_kwds={'extend':'both'}) 
    hindcasts[spec].boundary.plot(ax=curax,lw=0.5,color='k')
    plt.annotate(alpha[sct-1],(0.02,0.98),xycoords='axes fraction',ha='left',va='top')


#   Plot projections

def add_missing_country(ax,hindcast,curshp):
    #   Add gray missing countries
    missingi = [c for c in hindcast['country_id'].values 
                    if c not in curshp['country_id'].values]
    missingshp = hindcast.set_index('country_id').loc[missingi]
    missingshp['nan'] = 0
    missingshp.plot(ax=ax,color='lightgray')
    missingshp.boundary.plot(ax=ax,lw=0.5,color='k')
    return True


ctryshp['country_id'] = ctryshp['GID_0']
for si,spec in enumerate(specs):
    sct+=1
    curax = plt.subplot(5,len(specs),si+1+2*len(specs),projection=ccrs.Robinson())
    trackax.append(curax)
    curvar = (proj2050s[spec] - 1)*100

    curname = 'Change in '+map_name[plotvar]
    curvar.name = curname
    curshp = ctryshp.merge(curvar.to_dataframe(),on='country_id')
    curshp.plot(curname,ax=curax,#   transform=ccrs.PlateCarree(),
                vmin=-80,vmax=80,cmap='RdBu',
                legend=True,legend_kwds={'extend':'both'}) 
    curshp.boundary.plot(ax=curax,lw=0.5,color='k')
    add_missing_country(curax,hindcasts[spec],curshp)
    plt.annotate(alpha[sct-1],(0.02,0.98),xycoords='axes fraction',ha='left',va='top')


ctryshp['country_id'] = ctryshp['GID_0']
for si,spec in enumerate(specs):
    sct+=1
    curax = plt.subplot(5,len(specs),si+1+3*len(specs),projection=ccrs.Robinson())
    trackax.append(curax)
    curvar = (proj2050s_roll[spec] - 1)*100
    curname = 'Change in '+map_name[plotvar]
    curvar.name = curname
    curshp = ctryshp.merge(curvar.to_dataframe(),on='country_id')
    curshp.plot(curname,ax=curax,#   transform=ccrs.PlateCarree(),
                vmin=-80,vmax=80,cmap='RdBu',
                legend=True,legend_kwds={'extend':'both'}) 
    curshp.boundary.plot(ax=curax,lw=0.5,color='k')
    add_missing_country(curax,hindcasts[spec],curshp)
    plt.annotate(alpha[sct-1],(0.01,0.97),xycoords='axes fraction',ha='left',va='top')

#   Time series

for si,spec in enumerate(specs):
    sct+=1
    curax = plt.subplot(5,len(specs),si+1+4*len(specs))
    trackax.append(curax)
    hindtmp = hindcasts_full[specs[si]].reset_index().pivot(
                index='time',columns='country_id',values='predictions')
    hindtmp.plot(ax=curax,color='gray',alpha=0.5,lw=0.5,legend=False)

    projtmp = (projs[spec]/sspgdp['gdppc']).sel(year=range(2015,2051))
    projtmp = (projtmp - 1) * 100
    projtmp = projtmp.to_dataframe(name='proj').reset_index()
    projtmp = projtmp.pivot(index='year',columns='country_id',values='proj')
    projtmp.plot(ax=curax,color='cadetblue',alpha=0.5,lw=0.5,legend=False)

    projrolltmp = (projs_roll[spec]/sspgdp['gdppc']).sel(year=range(2015,2051))
    projrolltmp = (projrolltmp - 1) * 100
    projrolltmp = projrolltmp.to_dataframe(name='proj').reset_index()
    projrolltmp = projrolltmp.pivot(index='year',columns='country_id',values='proj')
    projrolltmp.plot(ax=curax,color='khaki',alpha=0.5,lw=0.5,legend=False)
    
    global_dGDP_roll = (projs_roll[spec].sum('country_id')/
                        sspgdp['gdppc'].sum('country_id')).to_dataframe(name='dgdp')
    global_dGDP_roll = ((global_dGDP_roll-1)*100).loc[range(2015,2051)]
    global_dGDP      = (projs[spec].sum('country_id')/
                        sspgdp['gdppc'].sum('country_id')).to_dataframe(name='dgdp')
    global_dGDP      = ((global_dGDP-1)*100).loc[range(2015,2051)]

    global_dGDP_roll.plot(ax=curax,color='darkgoldenrod',lw=1.5,legend=False)
    global_dGDP.plot(ax=curax,color='teal',lw=1.5,legend=False)

    curax.set_ylim((-80,50))
    if spec=='T':
        plt.ylabel('Change in \n$GDP_{pc}$ Growth [%]')
    plt.annotate(alpha[sct-1],(0.01,0.97),xycoords='axes fraction',ha='left',va='top')

h,l = curax.get_legend_handles_labels()
curax.legend(h[-2:],['Moving $\overline{T}_{ct}$','Fixed $\overline{T}_c$'])

trackax[len(specs)].annotate('Hindcast\n\n(2010)',(trackax[len(spec)].get_xlim()[0]*1.4,0),
                             rotation=90,fontsize=10,ha='center',va='center')
trackax[2*len(specs)].annotate('Projection\nfixed baseline ($\overline{T}_{c}$)\n(2010 to 2050)',
                (trackax[len(spec)].get_xlim()[0]*1.4,0),rotation=90,fontsize=10,ha='center',va='center')
trackax[3*len(specs)].annotate('Projection\nmoving baseline ($\overline{T}_{ct}$)\n(2010 to 2050)',
                (trackax[len(spec)].get_xlim()[0]*1.4,0),rotation=90,fontsize=10,ha='center',va='center')

plt.tight_layout()
plt.tight_layout()


plt.subplots_adjust(wspace=0.1)
for si,spec in enumerate(specs):
    pos = trackax[si+4*len(specs)].get_position().bounds
    trackax[si+4*len(specs)].set_position(pos*np.array([1,1,0.8,1]))

plt.savefig('../figures/figure8.png',dpi=600)



