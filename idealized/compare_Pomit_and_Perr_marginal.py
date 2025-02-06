"""
This file contains code to create Figure 5.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Patch

datomit  = pd.read_csv('../data/idealized_fits/spread_fits_PomitTests_usePFalse.csv',index_col=0)
datnomit = pd.read_csv('../data/idealized_fits/spread_fits_PomitTests_usePTrue.csv',index_col=0)

datomit  = datomit[datomit['feid']!='None']
datnomit = datnomit[datnomit['feid']!='None']

datomit['T2ratio']  = datomit['T2']  / datomit['T2_true']
datnomit['T2ratio'] = datnomit['T2'] / datnomit['T2_true']

datomit['Tmarginal_02'] = datomit['T'] + 2*datomit['T2']*0.2
datomit['Tmarginal_08'] = datomit['T'] + 2*datomit['T2']*0.8
datomit['Tmarginal_02_true'] = datomit['T_true'] + 2*datomit['T2_true']*0.2
datomit['Tmarginal_08_true'] = datomit['T_true'] + 2*datomit['T2_true']*0.8
datomit['Tmarginal_02_ratio'] = datomit['Tmarginal_02'] / datomit['Tmarginal_02_true']
datomit['Tmarginal_08_ratio'] = datomit['Tmarginal_08'] / datomit['Tmarginal_08_true']

datnomit['Tmarginal_02'] = datnomit['T'] + 2*datnomit['T2']*0.2
datnomit['Tmarginal_08'] = datnomit['T'] + 2*datnomit['T2']*0.8
datnomit['Tmarginal_02_true'] = datnomit['T_true'] + 2*datnomit['T2_true']*0.2
datnomit['Tmarginal_08_true'] = datnomit['T_true'] + 2*datnomit['T2_true']*0.8
datnomit['Tmarginal_02_ratio'] = datnomit['Tmarginal_02'] / datnomit['Tmarginal_02_true']
datnomit['Tmarginal_08_ratio'] = datnomit['Tmarginal_08'] / datnomit['Tmarginal_08_true']

datomit  = datomit.rename(columns={'w_TP_true':'T:P ratio'})
datnomit = datnomit.rename(columns={'w_TP_true':'T:P ratio'})

datomit['T:P ratio']  = datomit['T:P ratio'].astype(int)
datnomit['T:P ratio'] = datnomit['T:P ratio'].astype(int)

Terrstr = '0_std'
Perrstr = '1.0_spread'
i1 = (datomit['Terr']==Terrstr)&(datomit['Perr']==Perrstr)
i2 = (datnomit['Terr']==Terrstr)&(datnomit['Perr']==Perrstr)
datom = datomit[i1].groupby(['aggunit','feid','T:P ratio']).mean()
datno = datnomit[i2].groupby(['aggunit','feid','T:P ratio']).mean()

noP   = datom.reset_index().pivot(index='aggunit',columns='T:P ratio',values='Tmarginal_02_ratio')
noP.loc[['pixel','admin_id','country_id']]
noP.index = ['Pixel, omit P', 'Admin, omit P', 'Country, omit P']

withP = datno.reset_index().pivot(index='aggunit',columns='T:P ratio',values='Tmarginal_02_ratio')
withP.loc[['pixel','admin_id','country_id']]
withP.index = ['Pixel, imperfect P', 'Admin, imperfect P', 'Country, imperfect P']

bars02 = pd.concat([noP,withP]).iloc[[0,3,1,4,2,5]]
bars02.index = [i.replace(',','\n') for i in bars02.index]

noP   = datom.reset_index().pivot(index='aggunit',columns='T:P ratio',values='Tmarginal_08_ratio')
noP.loc[['pixel','admin_id','country_id']]
noP.index = ['Pixel, omit P', 'Admin, omit P', 'Country, omit P']

withP = datno.reset_index().pivot(index='aggunit',columns='T:P ratio',values='Tmarginal_08_ratio')
withP.loc[['pixel','admin_id','country_id']]
withP.index = ['Pixel, imperfect P', 'Admin, imperfect P', 'Country, imperfect P']

bars08 = pd.concat([noP,withP]).iloc[[0,3,1,4,2,5]]
bars08.index = [i.replace(',','\n') for i in bars08.index]

#   Make figure
Nbar = bars08.shape[1]
cols = [cm.Blues( (i+1)/(Nbar+1) ) for i in range(Nbar)][::-1]

plt.figure(figsize=(12,5))
ax1 = plt.subplot(1,2,1)
bars02.plot.bar(ax=ax1,color = cols).legend(loc='upper left',title='T:P ratio')
xlim = plt.gca().get_xlim()
plt.hlines(1,xmin=xlim[0],xmax=xlim[1],color='k',linestyle='--',alpha=0.5)
plt.ylim(0.95,1.2)
plt.ylabel('Fraction of true marginal T effect',fontsize=13)
plt.title('Cold (Normalized T=0.2)')
plt.gca().tick_params(axis='x', rotation=0)
plt.annotate('({})'.format('a'),(0.98,0.98),
                    xycoords='axes fraction',fontsize=14,ha='right',va='top')


ax2 = plt.subplot(1,2,2)
bars08.plot.bar(ax=ax2,color = cols).legend(loc='upper left',title='T:P ratio')
xlim = plt.gca().get_xlim()
plt.hlines(1,xmin=xlim[0],xmax=xlim[1],color='k',linestyle='--',alpha=0.5)
plt.ylim(0.95,1.2)
plt.ylabel('Fraction of true marginal T effect',fontsize=13)
plt.title('Warm (Normalized T=0.8)')
plt.gca().tick_params(axis='x', rotation=0)
plt.annotate('({})'.format('b'),(0.98,0.98),
                    xycoords='axes fraction',fontsize=14,ha='right',va='top')

plt.tight_layout()

#   Repeat with different colors
plt.figure(figsize=(12,5))
ax1 = plt.subplot(1,2,1)
colsB = [cm.Blues( (i+1)/(Nbar+1) ) for i in range(Nbar)][::-1]
colsG = [cm.Greens( (i+1)/(Nbar+1) ) for i in range(Nbar)][::-1]
colsP = [cm.Purples( (i+1)/(Nbar+1) ) for i in range(Nbar)][::-1]
colsGr= [cm.Greys( (i+1)/(Nbar+1) ) for i in range(Nbar)][::-1]
multicols = [colsB,colsB,colsG,colsG,colsP,colsP]
Ngrp = bars02.shape[1]
for i in range(len(bars02)):
    ax1.bar(x=i*(Ngrp+1)+np.arange(Ngrp),height=bars02.iloc[i],width=1,color=multicols[i])
ax1.set_xticks([1.5+(5*i) for i in range(bars02.shape[0])])
ax1.set_xticklabels(bars02.index)
xlim = ax1.get_xlim()
plt.hlines(1,xmin=xlim[0],xmax=xlim[1],color='k',linestyle='--',alpha=0.5)
ax1.set_xlim(xlim)
plt.ylim(0.95,1.2)
plt.ylabel('Fraction of true marginal T effect',fontsize=13)
plt.title('Cold (Normalized T=0.2)')
ax1.tick_params(axis='x', rotation=0)
plt.annotate('({})'.format('a'),(0.98,0.98),
                    xycoords='axes fraction',fontsize=14,ha='right',va='top')
legend_elements = [ Patch(facecolor=colsGr[i], edgecolor='k',
                    label=bars02.columns[i]) for i in range(Ngrp)]

ax1.legend(handles=legend_elements, loc='upper left',title='T:P ratio',fontsize=12,title_fontsize=12)

ax2 = plt.subplot(1,2,2)
Ngrp = bars08.shape[1]
for i in range(len(bars08)):
    ax2.bar(x=i*(Ngrp+1)+np.arange(Ngrp),height=bars08.iloc[i],width=1,color=multicols[i])
    #   bars02.plot.bar(ax=ax1,color = cols).legend(loc='upper left',title='T:P ratio')
ax2.set_xticks([1.5+(5*i) for i in range(bars08.shape[0])])
ax2.set_xticklabels(bars08.index)
xlim = ax2.get_xlim()
plt.hlines(1,xmin=xlim[0],xmax=xlim[1],color='k',linestyle='--',alpha=0.5)
ax2.set_xlim(xlim)
plt.ylim(0.95,1.2)
plt.ylabel('Fraction of true marginal T effect',fontsize=13)
plt.title('Warm (Normalized T=0.8)')
ax2.tick_params(axis='x', rotation=0)
plt.annotate('({})'.format('b'),(0.98,0.98),
                    xycoords='axes fraction',fontsize=14,ha='right',va='top')

plt.tight_layout()

plt.savefig('../figures/figure5.png',dpi=600)

