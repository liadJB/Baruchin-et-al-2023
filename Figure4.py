# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 12:36:44 2022

@author: LABadmin
"""

import numpy as np 
from matplotlib import pyplot as plt
import matplotlib as mpl
import random
import sklearn
import scipy as sp
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


from FileManagement import *
# from GlmModule import *
from DataProcessing import *
from Visualisation import *
from GeneralRunners import *
from PCA import *
from StatTests import *

saveDir = 'D:\\TaskProject\\EphysData\\'
if not os.path.isdir(saveDir):
    os.makedirs(saveDir)

plt.close('all')
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'Arial'

#%% Load Data
AllData,AllSubjects,AllSessions = LoadAllFiles('D:\\ProjectTask\\Ephys\\',ephys=True)
sessId,pId, nId = GetNeuronDetails(AllData,ephys=True)
r = np.load('D:\\Baruchin-et-al-2022\\\Data\\responsivenessEphys.npy')
nullProps1 = np.load('D:\\Figures\\Paper_Feedback\\Fitting\\Cumulative\\nulProps1_ephys.npy')
nullProps2 = np.load('D:\\Figures\\Paper_Feedback\\Fitting\\Cumulative\\nulProps2_ephys.npy')
#%% Panel A
plt.close('all')
sortby = r
d = [0, 400]
sigs = PrintPopulationResponsivenessEphys(AllData,sortBy = sortby,addLines=False,w = 40,depth=d)

#%% Panel B
plt.close('all')
sortby = r
d = [400.000001, 1000]
sigs = PrintPopulationResponsivenessEphys(AllData,sortBy = sortby,addLines=False,w = 40,depth=d)

#%% Panels C, D
def CreateNullIntervalPlot(ax,dist):
    ys = []
    f,axt = plt.subplots(1)
    # sns.histplot(data= dist.T,legend=False,common_norm=False,cumulative=True, stat='proportion',element='poly',fill=False,binrange=(-2,2),binwidth=0.3,ax=axt)
    sns.kdeplot(data=dist,legend=False,common_norm=False,common_grid = True,cumulative=True,ax=axt,clip=(-2,2),bw_adjust=0.1,cut=0)
    for i in range(len(axt.lines)):     
        line = axt.lines[i]
        x, y = line.get_data()
        y/=np.nanmax(y)
        ys.append(y)
        plt.close(f)
    ys = np.vstack(ys)
    
    # ax.plot(x,np.nanmean(ys,0),'grey')
    ax.fill_between(x,np.nanpercentile(ys, 2.5,axis=0),np.nanpercentile(ys, 97.5,axis=0),color = 'grey',alpha=0.4)
    return None

tups = np.vstack((sessId,pId,nId)).T
tups = tups[r[0,:]==1,:]
modelsBase = GetOneFunctionVarianceDifferentModels(dataDir = 'D:\\Figures\\OneFunction_ephys_raw\\',cutoff=0.01,shuffles= True,resp=tups)

plt.close('all')

depthList = -GetDepthFromList(AllData,modelsBase, 'D:\\Figures\\OneFunction_ephys_raw\\')
#Find the R for pupil and feedback
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



bins = np.arange(-1,1.1,0.03)
ymax = 80
##### Feedback Plotting #########
f,ax = plt.subplots(1)
ax.scatter(nfr[:,0]+nfr[:,1],nfr[:,0],s=30,c=['w'],edgecolors='grey')
ax.scatter(fr[:,0]+fr[:,1],fr[:,0],s=30,c=['k'],edgecolors='k')
ax.scatter(fpr[:,0]+fpr[:,2],fpr[:,0],s=30,c=['k'],edgecolors='k')
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.plot(np.arange(0,0.7,0.1),np.arange(0,0.7,0.1),'k--')
ax.set_title('R (gain) feedback')
ax.set_xlabel('Positive Feedback')
ax.set_ylabel('Negative Feedback')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
f.set_size_inches(8, 8)

# Extra histogram to put diagonally
f,ax = plt.subplots(1)
valsPos_ns = nullProps1[:,:,[0,4]]
valsPos_ns = valsPos_ns[:,:,1]/(0.5*(2*valsPos_ns[:,:,0]+valsPos_ns[:,:,1]))
valsPos = (frall[:,0]+frall[:,1]-frall[:,0])/(0.5*(frall[:,0]+frall[:,1]+frall[:,0]))

ff,axt = plt.subplots(1)
sns.kdeplot(data=[valsPos],palette=['black'],ax=axt,legend=False,common_norm=False,cumulative=True,clip=(-2,2),bw_adjust=0.1,cut=0)
line = axt.lines[0]
x, y = line.get_data()
y/=np.max(y)
plt.close(ff)
plt.plot(x,y,'k')
sorted_data = np.sort(valsPos)
y = np.arange(sorted_data.size)
y=y/np.max(y)
CreateNullIntervalPlot(ax,valsPos_ns)
plt.grid(False)
ax.set_xlabel('Respone Modulation (%)')
ax.set_ylabel('Proportion of Neurons')
ax.set_xlim(-2,2)
ax.set_ylim(0,1)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
f.set_size_inches(np.sqrt(8), np.sqrt(8))
ax.vlines(0,0,1,'k')

##### Pupil Plotting #########
f,ax = plt.subplots(1)
ax.scatter(npr[:,0]+npr[:,1],npr[:,0],s=30,c=['w'],edgecolors='grey')
ax.scatter(pr[:,0]+pr[:,1],pr[:,0],s=30,c=['k'],edgecolors='k')
ax.scatter(fpr[:,0]+fpr[:,1],fpr[:,0],s=30,c=['k'],edgecolors='k')
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.plot(np.arange(0,0.7,0.1),np.arange(0,0.7,0.1),'k--')
ax.set_title('R (gain) Pupil')
ax.set_xlabel('Large Pupil')
ax.set_ylabel('Small Pupil')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
f.set_size_inches(8, 8)

# Extra histogram to put diagonally
f,ax = plt.subplots(1)
valsPos_ns = nullProps2[:,:,[0,1]]
valsPos_ns = valsPos_ns[:,:,1]/(0.5*(2*valsPos_ns[:,:,0]+valsPos_ns[:,:,1]))

valsPos = (prall[:,1])/(0.5*(2*prall[:,0]+prall[:,1]))

ff,axt = plt.subplots(1)
sns.kdeplot(data=[valsPos],palette=['black'],ax=axt,legend=False,common_norm=False,cumulative=True,clip=(-2,2),bw_adjust=0.1,cut=0)
line = axt.lines[0]
x, y = line.get_data()
y/=np.max(y)
plt.close(ff)
plt.plot(x,y,'k')

CreateNullIntervalPlot(ax,valsPos_ns)
plt.grid(False)
ax.set_xlabel('Respone Modulation (%)')
ax.set_ylabel('Proportion of Neurons')
ax.set_xlim(-2,2)
ax.set_ylim(0,1)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
f.set_size_inches(np.sqrt(8), np.sqrt(8))
ax.vlines(0,0,1,'k')

#%% Panel E
plt.close('all')
tups = np.vstack((sessId,pId,nId)).T
tups = tups[r[0,:]==1,:]
modelsBase = GetOneFunctionVarianceDifferentModels(dataDir = 'D:\\Figures\\OneFunction_ephys_raw\\',flat=False,cutoff=0.01,shuffles= True,resp=tups)
fits = modelsBase.reset_index(drop = True)
dsp = GetDepthFromList(AllData,fits, 'D:\\Figures\\OneFunction_ephys_raw\\')
print('P')
fits = modelsBase[modelsBase['bestFit']=='P'].reset_index(drop = True)
dsp = GetDepthFromList(AllData,fits,'D:\\Figures\\OneFunction_ephys_raw\\')
print('PF')
fits = modelsBase[modelsBase['bestFit']=='PF'].reset_index(drop = True)
dspf = GetDepthFromList(AllData,fits,'D:\\Figures\\OneFunction_ephys_raw\\')
print('G')
fits = modelsBase[modelsBase['bestFit']=='G'].reset_index(drop = True)
dsg = GetDepthFromList(AllData,fits,'D:\\Figures\\OneFunction_ephys_raw\\')
print('F')
fits = modelsBase[modelsBase['bestFit']=='F'].reset_index(drop = True)
dsf = GetDepthFromList(AllData,fits,'D:\\Figures\\OneFunction_ephys_raw\\')
print('None')
fits = modelsBase[modelsBase['bestFit']=='None'].reset_index(drop = True)
dsn = GetDepthFromList(AllData,fits,'D:\\Figures\\OneFunction_ephys_raw\\')

fits = modelsBase[modelsBase['bestFit']=='C'].reset_index(drop = True)
dsc = GetDepthFromList(AllData,fits,'D:\\Figures\\OneFunction_ephys_raw\\')

dsph,_ = np.histogram(dsp,range(-1000,0,333))
dsfh,_ = np.histogram(dsf,range(-1000,0,333))
dsgh,_ = np.histogram(dsg,range(-1000,0,333))
dsch,_ = np.histogram(dsc,range(-1000,0,333))

f,ax = plt.subplots(1)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
sns.violinplot(data=[dsn,dsf,dsp,dsg,dsc],ax=ax,color='grey')
sns.swarmplot(data=[dsn,dsf,dsp,dsg,dsc],color='white',edgecolor='gray',ax=ax)
ax.set_xticks(range(0,5))
ax.set_xticklabels(['None','Previous Feedback','Pupil','Go','Current Feedback'])
ax.set_ylabel('Depth relative to SC surgace (mm)')
ax.hlines(-400,-100,100,'k',linestyles='--')
