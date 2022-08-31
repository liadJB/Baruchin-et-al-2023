# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 10:09:24 2022

@author: LABadmin
"""
#%% Load Data
import numpy as np 
from matplotlib import pyplot as plt
import random
import sklearn
import seaborn as sns
import scipy as sp
from matplotlib import rc
import matplotlib.ticker as mtick
import matplotlib as mpl

from scipy import signal
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer   
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import ElasticNetCV
from sklearn.datasets import make_regression
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split

import os
import glob
import pickle

import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.tsa.stattools import acf
from statsmodels.stats.multicomp import pairwise_tukeyhsd


# from FileManagement import *
# from GlmModule import *
from DataProcessing import *
from Visualisation import *
from GeneralRunners import *
from PCA import *
from Analysis import *
from Controls import *

saveDir = 'C:\\TaskProject\\2pData\\'
if not os.path.isdir(saveDir):
    os.makedirs(saveDir)    
    
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'Arial'
#%% Load Data 2p 
AllData,AllSubjects,AllSessions = LoadAllFiles('D:\\task_2p\\')
sessId,nId = GetNeuronDetails(AllData)
tups = np.vstack((sessId,nId)).T
tups = list(tuple(map(tuple,tups)))
r = np.load('D:\\Baruchin2022Code\\Data\\responsiveness2p.npy')
modelsBase = GetOneFunctionVarianceDifferentModels(dataDir = 'D:\\Figures\\OneFunction_raw\\',cutoff=0.00001,shuffles= True,resp=tups_)

#%% Panel A
s = 1
d = AllData[s]
pupil = d['pupil']
pt = d['pupilTimes']
ft = d['feedbackTimes']
feedbackType = d['feedbackType']
maxPupil = np.nanmedian(pupil)
f,ax = plt.subplots(2,1,sharex=(True))
ax[1].plot(pt,pupil,'k')
ax[0].vlines(ft[feedbackType==0],0,1,'r')
ax[0].vlines(ft[feedbackType==1],0,1,'g')

#%%  Panel B
GetPupilSize(AllData,'feedback', plot = True)

#%% Panel C
f1 = np.array(modelsBase[modelsBase['bestFit']=='F']['fit'])
fp1 = np.array(modelsBase[modelsBase['bestFit']=='PF']['fit'])
p1 = np.array(modelsBase[modelsBase['bestFit']=='P']['fit'])
nf1 = np.array(modelsBase[modelsBase['bestFit']!='F']['fit'])
np1 = np.array(modelsBase[modelsBase['bestFit']!='P']['fit'])
nfp1 = np.array(modelsBase[modelsBase['bestFit']!='PF']['fit'])
g1 = np.array(modelsBase[modelsBase['bestFit']=='G']['fit'])
c1 = np.array(modelsBase[modelsBase['bestFit']=='C']['fit'])
ng1 = np.array(modelsBase[modelsBase['bestFit']!='G']['fit'])
nc1 = np.array(modelsBase[modelsBase['bestFit']!='C']['fit'])
gc1 = np.array(modelsBase[modelsBase['bestFit']!='GC']['fit'])
allFit = np.array(modelsBase['fit'])


fr= GetRFromFitList(list(f1),'F')
fpr= GetRFromFitList(list(fp1),'PF')
nfr = GetRFromFitList(list(nf1),'F')
pr = GetRFromFitList(list(p1),'P')
npr = GetRFromFitList(list(np1),'P')
nfpr = GetRFromFitList(list(nfp1),'PF')
gr = GetRFromFitList(list(g1),'G')
ngr = GetRFromFitList(list(ng1),'G')
cr = GetRFromFitList(list(c1),'C')
ncr = GetRFromFitList(list(nc1),'C')

frall= GetRFromFitList(list(allFit),'F')
prall = GetRFromFitList(list(allFit),'P')
grall = GetRFromFitList(list(allFit),'G')
crall = GetRFromFitList(list(allFit),'C')




##### Pupil Feedback Plotting #########
f,ax = plt.subplots(1)
a = (nfpr[:,1])/(0.5*(2*nfpr[:,0]+nfpr[:,1]))
b = (nfpr[:,2])/(0.5*(2*nfpr[:,0]+nfpr[:,2]))
ax.scatter(a,b,s=30,c=(0.6,0.6,0.6))
a = (fpr[:,1])/(0.5*(2*fpr[:,0]+fpr[:,1]))
b = (fpr[:,2])/(0.5*(2*fpr[:,0]+fpr[:,2]))

ax.scatter(a,b,s=30,c=['k'])


ax.set_aspect('equal', adjustable='box')

ax.hlines(0,-2,2,'k',linestyles='dashed')
ax.vlines(0,-2,2,'k',linestyles='dashed')
ax.set_title('Pupil gain vs. Feeback gain')
ax.set_xlabel('Pupil gain')
ax.set_ylabel('Feedback gain')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

#%% Panel D-F
plt.close('all')
lck = ShowLickingForState(AllData,influence = 'feedback')
percPos = []
percNeg = []

# lck__ = lck.copy()
# for dd in range(2,6):
#     del lck[dd]
# lck = np.array[[0,5,6,7,8,9,10,11,12,13,14,15]]
indPrePos = 0
indPreNeg = 2
indPostPos = 1
indPostNeg = 3

for l in lck:
    percPos.append(np.sum((l[indPostPos]>0)/len(l[indPrePos]))*100)
    percNeg.append(np.sum((l[indPostNeg]>0)/len(l[1]))*100)
    
NumPos = []
NumNeg = []
NumPospre = []
NumNegpre = []
for l in lck:
    NumPos.append(np.mean(l[indPostPos]))
    NumNeg.append(np.mean(l[indPostNeg]))
    NumPospre.append(np.mean(l[indPrePos]))
    NumNegpre.append(np.mean(l[indPreNeg]))
    
percPos = np.array(percPos)
percNeg = np.array(percNeg)

NumPos = np.array(NumPos)
NumNeg = np.array(NumNeg)
NumPospre = np.array(NumPospre)
NumNegpre = np.array(NumNegpre)

df = pd.DataFrame({'Pupil': np.hstack((np.repeat(['Large'], 16*2),(np.repeat(['Small'], 16*2)))),
                   'Period': np.tile(np.hstack((np.repeat(['pre'],16),np.repeat(['post'], 16))),(1,2))[0,:],
                   'licks': np.hstack((NumPospre,NumPos,NumNegpre,NumNeg))})

model = ols('licks ~ C(Pupil) + C(Period) + C(Pupil):C(Period)', data=df).fit()
anov = sm.stats.anova_lm(model, typ=2)


a = sns.catplot(x="Period", y="licks", hue="Pupil", data=df, kind='point',palette = ['b','r'])
plt.xlabel('Licks (Hz)')
plt.tight_layout()

#%% panel H
tups = np.vstack((sessId,nId)).T
tups = tups[r[0,:]==1,:]
prog, progCon,pup,fb = GetResponseProgression(AllData,fitTime = 0.5)
ids = np.vstack(np.sort(np.array(modelsBase['Id'])))

maxLen = 0
for i in range(len(prog)):
    if prog[i].shape[0]>maxLen:
        maxLen = prog[i].shape[0]

progress = np.zeros((ids.shape[0],maxLen))*np.nan

for iid in range(ids.shape[0]):    
    sess = ids[iid,0]-1
    n = ids[iid,1]    
    p = prog[sess][:,n]
    progress[iid,:len(p)] = p
    

def CreateCI(ac,last = 50,reps = 500):    
    ci = np.zeros((ac.shape[0],2))
    sig = np.random.choice(ac[:],reps)
    for t in range(ac.shape[0]):      
        # sig = np.roll(sig,1)
        
        ci[t,0] = np.percentile(sig,2.5)
        ci[t,1] = np.percentile(sig,97.5)      
    
    return ci
    
    
plt.close('all')

acsTauFitTotal = []
exceedCi = []  

iid = 7     
prog1 = progress[np.where(ids[:,0]==iid)[0],0:len(pup[iid-1])]
acs = np.zeros((prog1.shape[0],prog1.shape[1]*2-1))
acsTauFit = np.zeros((prog1.shape[0],1))
corRange = np.array(range(-prog1.shape[1]+1,prog1.shape[1]))    
               
for pp in range(prog1.shape[0]):
    acs[pp,:] = sp.signal.correlate(prog1[pp,:],prog1[pp,:])
    af,ci = acf(prog1[pp,:],alpha=0.05,nlags = prog1[pp,:].shape[0])        
    ci = CreateCI(af)     
    Inds = np.where((af>ci[:,1]))[0]
    Inds = Inds[np.where(np.diff(Inds,prepend=True)<=1)]
    firstStop = np.where(np.diff(Inds,prepend=True)>1)[0]
    if (len(firstStop)>0):            
        Inds = Inds[:firstStop[0]]
    
    exceedCi.append(Inds)
    fitPart = acs[pp,corRange>=0]
    
f,ax = plt.subplots(1,1,sharex = True)    
acsTauFitTotal.append(np.vstack(results))
f.suptitle('Session '+str(iid))
ax.plot(range(-prog1.shape[1]+1,prog1.shape[1]),np.nanmean(acs,0))
ci = CreateCI(np.nanmean(acs,0).T)
plt.fill_between(range(-prog1.shape[1]+1,prog1.shape[1]), ci[:,0], ci[:,1],alpha=0.2,facecolor='grey')
# ax[1].set_xlim(-20,20)
ax.set_title('Average Trace')
ax.set_xlabel('lag (trials)')    
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

#%% Panel I
lastInd = []  
exceedCi = []

tups = np.vstack((sessId,nId)).T
tups = tups[r[0,:]==1,:]
ids = np.vstack(np.sort(np.array(modelsBase['Id'])))


for iid in np.unique(ids[:,0]):#range(len(np.unique(ids[:,0]))):    
    prog1 = progress[np.where(ids[:,0]==iid)[0],0:len(pup[iid-1])]
    acs = np.zeros((prog1.shape[0],prog1.shape[1]*2-1))
    acsTauFit = np.zeros((prog1.shape[0],1))
    corRange = np.array(range(-prog1.shape[1]+1,prog1.shape[1]))   
    
            
    for pp in range(prog1.shape[0]):
        acs[pp,:] = sp.signal.correlate(prog1[pp,:],prog1[pp,:])
        af,ci = acf(prog1[pp,:],alpha=0.05,nlags = prog1[pp,:].shape[0])
        # ci =CreateCIShift(prog1[pp,:])
        ci = CreateCI(af)
        # Find where correlation exceeds 95% far confidence interval
        Inds = np.where((af>ci[:,1]))[0]
        Inds = Inds[np.where(np.diff(Inds,prepend=True)<=1)]
        firstStop = np.where(np.diff(Inds,prepend=True)>1)[0]
        Inds = np.where((af>ci[:,1]))[0]
        Inds = Inds[np.where(np.diff(Inds,prepend=True)<=1)]
        firstStop = np.where(np.diff(Inds,prepend=True)>1)[0]
        if (len(firstStop)>0):            
            Inds = Inds[:firstStop[0]]
        # Inds = Inds[Inds<50]
        exceedCi.append(Inds)
        
last = []
for i in range(len(exceedCi)):
    if (len(exceedCi[i])>0):
        last.append(exceedCi[i][-1])
    else:
        last.append(0)
sns.histplot(last,bins = 9)
plt.xlim(0,4)

         
#%% Panels J,K
plt.close('all')
tups = np.vstack((sessId,nId)).T
tups = tups[r[0,:]==1,:]

modelsBase = GetOneFunctionVarianceDifferentModels(dataDir = 'D:\\Figures\\OneFunction_raw\\',cutoff=0.01,shuffles= True,resp=tups).sort_index()

modelsLick = GetOneFunctionVarianceDifferentModels(dataDir = 'D:\\Figures\\OneFunction_raw_NoStimLick\\',cutoff=0.01,shuffles= True,resp=tups).sort_index()

# run multiple times for different conditions (1 - was sig then not, 2- was not sig but now is, 3 - stayed sig)

f,ax = plt.subplots(1,2)

colors = [(0.8,0.8,0.8),(0.5,0.5,0.5),(0,0,0)]
sF = []
sP = []
for cond in [1,2,3]:
    if (cond==3):
        lickSharedFi = pd.Series(list(set(modelsBase[modelsBase.bestFit=='F']['Id']).intersection(set(modelsLick[modelsLick.bestFit=='F']['Id'])))).sort_index()
        lickSharedPFi = list(set(modelsBase[modelsBase.bestFit=='PF']['Id']).intersection(set(modelsLick[modelsLick.bestFit=='PF']['Id'])))
        lickSharedPi = list(set(modelsBase[modelsBase.bestFit=='P']['Id']).intersection(set(modelsLick[modelsLick.bestFit=='P']['Id'])))
    if (cond == 2):
        lickSharedFi = pd.Series(list(set(modelsBase[modelsBase.bestFit!='F']['Id']).intersection(set(modelsLick[modelsLick.bestFit=='F']['Id'])))).sort_index()
        lickSharedPFi = list(set(modelsBase[modelsBase.bestFit!='PF']['Id']).intersection(set(modelsLick[modelsLick.bestFit=='PF']['Id'])))
        lickSharedPi = list(set(modelsBase[modelsBase.bestFit!='P']['Id']).intersection(set(modelsLick[modelsLick.bestFit=='P']['Id'])))
    if (cond == 1):
        lickSharedFi = pd.Series(list(set(modelsBase[modelsBase.bestFit=='F']['Id']).intersection(set(modelsLick[modelsLick.bestFit!='F']['Id'])))).sort_index()
        lickSharedPFi = list(set(modelsBase[modelsBase.bestFit=='PF']['Id']).intersection(set(modelsLick[modelsLick.bestFit!='PF']['Id'])))
        lickSharedPi = list(set(modelsBase[modelsBase.bestFit=='P']['Id']).intersection(set(modelsLick[modelsLick.bestFit!='P']['Id'])))
    sharedF = []
    sharedP = []
    
    for share in lickSharedFi:
        fitBase = modelsBase[modelsBase.Id.reset_index(drop=True)==share]['fit'].item()['fitInclude'][4,:]
        fitLick = modelsLick[modelsLick.Id.reset_index(drop=True)==share]['fit'].item()['fitInclude'][4,:]
        fitBase = fitBase[4]/(0.5*(2*fitBase[0]+fitBase[4]))
        fitLick = fitLick[4]/(0.5*(2*fitLick[0]+fitLick[4]))
        sharedF.append((fitBase,fitLick)) 
    
    for share in lickSharedPi:
        fitBase = modelsBase[modelsBase.Id.reset_index(drop=True)==share]['fit'].item()['fitInclude'][1,:]
        fitLick = modelsLick[modelsLick.Id.reset_index(drop=True)==share]['fit'].item()['fitInclude'][1,:]
        fitBase = fitBase[1]/(0.5*(2*fitBase[0]+fitBase[1]))
        fitLick = fitLick[1]/(0.5*(2*fitLick[0]+fitLick[1]))
        sharedP.append((fitBase,fitLick))
        
    sharedF = np.vstack(sharedF).T
    sharedP = np.vstack(sharedP).T
    
    sF.append(sharedF)
    sP.append(sharedP)
    
    ax[0].scatter(*sharedF,s=30,color=colors[cond-1])
    ax[0].set_aspect('equal', adjustable='box')
    ax[0].set_xlim(-2,2)
    ax[0].set_ylim(-2,2)
    ax[0].plot(np.arange(-2,2,0.1),np.arange(-2,2,0.1),'k--')
    ax[0].set_title('R (gain) feedback')
    ax[0].set_xlabel('Baseline R gain')
    ax[0].set_ylabel('No Lick Baseline R gain')
    ax[0].spines['right'].set_visible(False)
    ax[0].spines['top'].set_visible(False)
    f.set_size_inches(8, 8)
    
    ax[1].scatter(*sharedP,s=30,color=colors[cond-1])
    ax[1].set_aspect('equal', adjustable='box')
    ax[1].set_xlim(-2,2)
    ax[1].set_ylim(-2,2)
    ax[1].plot(np.arange(-2,2,0.1),np.arange(-2,2,0.1),'k--')
    ax[1].set_title('R (gain) Pupil')
    ax[1].set_xlabel('Baseline R gain')
    ax[1].set_ylabel('No Lick Baseline R gain')
    ax[1].spines['right'].set_visible(False)
    ax[1].spines['top'].set_visible(False)
    f.set_size_inches(8, 8)
f,axx = plt.subplots(1)
axx.bar([1,2,3],[1,1,1],color=colors)
axx.set_title('1 - sig/not 2 - not/sig 3 - sig/sig')

#%% Panels L,M
plt.close('all')
tups = np.vstack((sessId,nId)).T
tups = tups[r[0,:]==1,:]

modelsBase = GetOneFunctionVarianceDifferentModels(dataDir = 'D:\\Figures\\OneFunction_raw\\',cutoff=0.01,shuffles= True,resp=tups).sort_index()

modelsLick = GetOneFunctionVarianceDifferentModels(dataDir = 'D:\\Figures\\OneFunction_rawDrift\\',cutoff=0.01,shuffles= True,resp=tups).sort_index()

# run multiple times for different conditions (1 - was sig then not, 2- was not sig but now is, 3 - stayed sig)

f,ax = plt.subplots(1,2)

colors = [(0.8,0.8,0.8),(0.5,0.5,0.5),(0,0,0)]
sF = []
sP = []
for cond in [1,2,3]:
    if (cond==3):
        lickSharedFi = pd.Series(list(set(modelsBase[modelsBase.bestFit=='F']['Id']).intersection(set(modelsLick[modelsLick.bestFit=='F']['Id'])))).sort_index()
        lickSharedPFi = list(set(modelsBase[modelsBase.bestFit=='PF']['Id']).intersection(set(modelsLick[modelsLick.bestFit=='PF']['Id'])))
        lickSharedPi = list(set(modelsBase[modelsBase.bestFit=='P']['Id']).intersection(set(modelsLick[modelsLick.bestFit=='P']['Id'])))
    if (cond == 2):
        lickSharedFi = pd.Series(list(set(modelsBase[modelsBase.bestFit!='F']['Id']).intersection(set(modelsLick[modelsLick.bestFit=='F']['Id'])))).sort_index()
        lickSharedPFi = list(set(modelsBase[modelsBase.bestFit!='PF']['Id']).intersection(set(modelsLick[modelsLick.bestFit=='PF']['Id'])))
        lickSharedPi = list(set(modelsBase[modelsBase.bestFit!='P']['Id']).intersection(set(modelsLick[modelsLick.bestFit=='P']['Id'])))
    if (cond == 1):
        lickSharedFi = pd.Series(list(set(modelsBase[modelsBase.bestFit=='F']['Id']).intersection(set(modelsLick[modelsLick.bestFit!='F']['Id'])))).sort_index()
        lickSharedPFi = list(set(modelsBase[modelsBase.bestFit=='PF']['Id']).intersection(set(modelsLick[modelsLick.bestFit!='PF']['Id'])))
        lickSharedPi = list(set(modelsBase[modelsBase.bestFit=='P']['Id']).intersection(set(modelsLick[modelsLick.bestFit!='P']['Id'])))
    sharedF = []
    sharedP = []
    
    for share in lickSharedFi:
        fitBase = modelsBase[modelsBase.Id.reset_index(drop=True)==share]['fit'].item()['fitInclude'][4,:]
        fitLick = modelsLick[modelsLick.Id.reset_index(drop=True)==share]['fit'].item()['fitInclude'][4,:]
        fitBase = fitBase[4]/(0.5*(2*fitBase[0]+fitBase[4]))
        fitLick = fitLick[4]/(0.5*(2*fitLick[0]+fitLick[4]))
        sharedF.append((fitBase,fitLick)) 
    
    for share in lickSharedPi:
        fitBase = modelsBase[modelsBase.Id.reset_index(drop=True)==share]['fit'].item()['fitInclude'][1,:]
        fitLick = modelsLick[modelsLick.Id.reset_index(drop=True)==share]['fit'].item()['fitInclude'][1,:]
        fitBase = fitBase[1]/(0.5*(2*fitBase[0]+fitBase[1]))
        fitLick = fitLick[1]/(0.5*(2*fitLick[0]+fitLick[1]))
        sharedP.append((fitBase,fitLick))
        
    sharedF = np.vstack(sharedF).T
    sharedP = np.vstack(sharedP).T
    
    sF.append(sharedF)
    sP.append(sharedP)
    
    ax[0].scatter(*sharedF,s=30,color=colors[cond-1])
    ax[0].set_aspect('equal', adjustable='box')
    ax[0].set_xlim(-2,2)
    ax[0].set_ylim(-2,2)
    ax[0].plot(np.arange(-2,2,0.1),np.arange(-2,2,0.1),'k--')
    ax[0].set_title('R (gain) feedback')
    ax[0].set_xlabel('Baseline R gain')
    ax[0].set_ylabel('No Lick Baseline R gain')
    ax[0].spines['right'].set_visible(False)
    ax[0].spines['top'].set_visible(False)
    f.set_size_inches(8, 8)
    
    ax[1].scatter(*sharedP,s=30,color=colors[cond-1])
    ax[1].set_aspect('equal', adjustable='box')
    ax[1].set_xlim(-2,2)
    ax[1].set_ylim(-2,2)
    ax[1].plot(np.arange(-2,2,0.1),np.arange(-2,2,0.1),'k--')
    ax[1].set_title('R (gain) Pupil')
    ax[1].set_xlabel('Baseline R gain')
    ax[1].set_ylabel('No Lick Baseline R gain')
    ax[1].spines['right'].set_visible(False)
    ax[1].spines['top'].set_visible(False)
    f.set_size_inches(8, 8)
f,axx = plt.subplots(1)
axx.bar([1,2,3],[1,1,1],color=colors)
axx.set_title('1 - sig/not 2 - not/sig 3 - sig/sig')  
