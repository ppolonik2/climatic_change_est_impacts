import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import pandas as pd
import pdb

def rangenorm(allx,x):
    return (x-np.min(allx))/(np.max(allx)-np.min(allx))

def rangenormhalf(allx,x):
    return (x-0.5*np.min(allx))/(np.max(allx)-np.min(allx))

def unrangenorm(allx,xn):
    return xn * (np.max(allx)-np.min(allx)) + np.min(allx)

def outfunc(c1,c2,c3,c4,c5,T,P,inter=0):
    out = c1*T + c2*T*T + c3*P + c4*P*P + c5 + inter*P*T
    return out

def calc_params(Toptf,Poptf,w):
    c2a = 1/(-Toptf*Toptf - 1/w*Poptf*Poptf)
    c2b = 1/(-Toptf*Toptf - 1/w*Poptf*Poptf + 2*Toptf - 1)
    c2c = 1/(-Toptf*Toptf - 1/w*Poptf*Poptf + 2/w*Poptf - 1/w)
    c2d = 1/(-Toptf*Toptf - 1/w*Poptf*Poptf + 2*Toptf - 1 + 2/w*Poptf - 1/w)
    
    c2s = [c2a,c2b,c2c,c2d]
    
    outcomems = []
    all_c_combos = []
    T01,P01 = np.meshgrid([0,1],[0,1])
    for i,c2 in enumerate(c2s):
        c1 = -2*c2*Toptf
        c4 = c2/w
        c3 = -2*c4*Poptf
        c5 = 1 - c1*Toptf - c2*Toptf*Toptf - c3*Poptf - c4*Poptf*Poptf
    
        if i==0:
            eq5 = c5
        elif i==1:
            eq5 = c1+c2+c5
        elif i==2:
            eq5 = c3+c4+c5
        elif i==3:
            eq5 = c1+c2+c3+c4+c5
     
        outcomems.append(outfunc(c1,c2,c3,c4,c5,T01,P01))
        all_c_combos.append([c1,c2,c3,c4,c5])
    
    outcomems = np.array(outcomems)
    outcomemins = outcomems.min(-1).min(-1)
    usec2 = np.argmin(np.abs(outcomemins))
    
    outcomem = outcomems[usec2]
    cs = all_c_combos[usec2]
    return cs

def quadout(Topt,Popt,w,T=[],P=[]):
    if len(T)==0:
        T = np.arange(-5,35)
    if len(P)==0:
        P = np.arange(0,4,0.1) #   Assuming units are m
    Tm, Pm = np.meshgrid(T,P)
    Tf = rangenorm(T,T)
    Pf = rangenorm(P,P)
    
    Toptf = rangenorm(T,Topt)
    Poptf = rangenorm(P,Popt)
    
    cs = calc_params(Toptf,Poptf,w)
    outcomem = outfunc(cs[0],cs[1],cs[2],cs[3],cs[4],rangenorm(T,Tm),rangenorm(P,Pm))
        
    yT = cs[0]*Tf + cs[1]*Tf*Tf
    yP = cs[2]*Pf + cs[3]*Pf*Pf

    return T, P, Tm, Pm, Toptf, Poptf, Tf, Pf, yT, yP, outcomem, cs

#   def lininterout()

def plotout(T, P, Tm, Pm, Toptf, Poptf, Tf, Pf, yT, yP, w, outcomem):
    plt.figure(figsize=(10,5));
    plt.subplot(1,2,1)
    plt.plot(Tf,yT,'r',label='T')
    plt.plot(Pf,yP,'b',label='P')
    ax = plt.gca()
    ax.set_xlim([0,1])
    
    #   Make funny labels at the top
    axT = ax.twiny()
    xticks = axT.get_xticks()
    Tticks = unrangenorm(T,xticks)
    Pticks = unrangenorm(P,xticks)
    label = ['T: {:.1f}\nP: {:.1f}'.format(t,p) for t,p in zip(Tticks,Pticks)]
    axT.set_xticklabels(label)
    
    #   Add Topt and Popt lines
    ylim = ax.get_ylim()
    ax.plot([Toptf,Toptf],ylim,'--r',alpha=0.5)
    ax.plot([Poptf,Poptf],ylim,'--b',alpha=0.5)
    Toptfstr = '{:.2f}C'.format(Toptf)
    Poptfstr = '{:.2f}m'.format(Poptf)
    ax.text(Toptf+0.01,ylim[0]+0.05*np.diff(ylim),'T$_{optf}$='+Toptfstr,color='r',fontsize=11)
    ax.text(Poptf+0.01,ylim[0]+0.11*np.diff(ylim),'P$_{optf}$='+Poptfstr,color='b',fontsize=11)
    ax.set_ylabel('Relative Outcome')
    ax.set_xlabel('Normalized T or P')
    
    plt.title('w = '+str(w))
    
    #   Check slopes
    Tmini = np.argmin(np.abs(Tf-Toptf))
    Tslope = (yT[Tmini+1]-yT[Tmini])/(Tf[Tmini+1]-Tf[Tmini])
    Pmini = np.argmin(np.abs(Pf-Poptf))
    Pslope = (yP[Pmini+1]-yP[Pmini])/(Pf[Pmini+1]-Pf[Pmini])
    w_calc = Tslope/Pslope
    
    #   Move on to 2D version
    #   as function of T and P, what is the outcome?
    #       Make a few and then make one with contours where Topt and Popt are changed
    plt.subplot(1,2,2)
    cbar = plt.contourf(Tm,Pm,outcomem,
                        np.arange(0,1.01,.1),ticks=np.arange(0,1.01,0.2))
    plt.colorbar(label='Outcome')
    
    plt.xlabel('T [C]')
    plt.ylabel('P [m]')
    
    plt.tight_layout()

def create_data(Nt,Nx,cov,Topt,Popt,w,Tstderr,Pstderr,Trange,Prange):
    """
    modelrun(Nt,Nx,use_FE,cov,Pcoef,Tcoef,Pstderr,Tstderr,outcome_const):
    """
    #   Initialize
    combnames = ['P','T','Perr','Terr','P2','T2','Perr2','Terr2',
                 'Pf','Tf','Perrf','Terrf','P2f','T2f','Perr2','Terr2','Perr2f','Terr2f',
                 'outcome','outcomeerr','fe1','fe2']
    data = pd.DataFrame(columns=combnames)

    Trange, Prange, Tm, Pm, Toptf, Poptf, Tf, Pf, yT, yP, outcomem, cs = quadout(Topt,Popt,float(w),
                                                                          Trange,Prange)

    #   Create data
    for i in range(Nx):
        #   Generate random normal data with covariance
        T, P = np.random.multivariate_normal(mean=[0,0],cov=[[1,cov],[cov,1]],size=Nt).T
        T = T*(np.max(Trange)-np.min(Trange))/3+np.mean(Trange)
        P = P*(np.max(Prange)-np.min(Prange))/3+np.mean(Prange)

        #   Check whether any data fall outside of allowable range
        outofbounds = ((T<np.min(Trange)) | (T>np.max(Trange)) | 
                       (P<np.min(Prange)) | (P>np.max(Prange)))
        #   If they do, re-write those and repeat until they're gone
        while np.sum(outofbounds)>0:
            Tnew, Pnew = np.random.multivariate_normal(mean=[0,0],
                             cov=[[1,cov],[cov,1]],size=np.sum(outofbounds)).T
            Tnew = Tnew*(np.max(Trange)-np.min(Trange))/3+np.mean(Trange)
            Pnew = Pnew*(np.max(Prange)-np.min(Prange))/3+np.mean(Prange)
            T[outofbounds] = Tnew
            P[outofbounds] = Pnew
            outofbounds = ((T<np.min(Trange)) | (T>np.max(Trange)) | 
                           (P<np.min(Prange)) | (P>np.max(Prange)))

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
            T = T*1.

        Tf = rangenorm(Trange,T)
        Pf = rangenorm(Prange,P)
        T2 = T*T
        P2 = P*P
        T2f = Tf*Tf
        P2f = Pf*Pf
        Terr2 = Terr*Terr
        Perr2 = Perr*Perr
        Terrf = rangenorm(Trange,Terr)
        Perrf = rangenorm(Prange,Perr)
        Terr2f = Terrf*Terrf
        Perr2f = Perrf*Perrf
        outcome = outfunc(cs[0],cs[1],cs[2],cs[3],cs[4],Tf,Pf)
        outcomeerr = outfunc(cs[0],cs[1],cs[2],cs[3],cs[4],Terrf,Perrf)

        combined = np.array([P,T,Perr,Terr,
                             P2,T2,Perr2,Terr2,
                             Pf,Tf,Perrf,Terrf,P2f,T2f,
                             Perr2,Terr2,Perr2f,Terr2f,
                             outcome,outcomeerr,fe1,fe2]).T

        data = pd.concat([data,pd.DataFrame(combined,columns=combnames)])

    data = data.reset_index()
    return data, cs

def get_coefs(formula,data,return_errors=False):
    mod = smf.ols(formula,data)
    res = mod.fit()

    if not return_errors:
        return res.params
    else:
        return res.params, res.bse

def get_coefs_w(formula,data,wtcol,return_errors=False):
    mod = smf.wls(formula,data,weights=data[wtcol].values)
    res = mod.fit()

    if not return_errors:
        return res.params
    else:
        return res.params, res.bse

def outcome_from_coefs(Tmat,Pmat,coefs,formula,intopt=True):
    outcome = np.zeros_like(Tmat)

    for cname,cval in coefs.items():

        if 'T' in cname:
            if '2' in cname:
                outcome += cval*Tmat*Tmat
            else:
                outcome += cval*Tmat
        elif 'P' in cname:
            if '2' in cname:
                outcome += cval*Pmat*Pmat
            else:
                outcome += cval*Pmat
        elif (cname=='Intercept')&(intopt):
            outcome += cval

    return outcome
    
