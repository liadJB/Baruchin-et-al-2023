# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 09:15:06 2022

@author: LABadmin
"""

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
from StatTests import *
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

#%% Panel B
plt.close('all')
GetChoiPerAnimal(AllData,plot = 1)


#%% Panel C
GetReactionTime(AllData,plot = 1)

#%% Panel D
# load neuronal responsivness Factor X Neuron
# Order : Contra stim, ipsi stim, cue, contra move, ipsi move, pos feedback, neg feedback, lick
sigs = PrintPopulationResponsiveness(AllData,sortBy = r)

#%% Panel E
sigs = PrintPopulationResponsiveness(AllData,sortBy = r,addLines=False,focusOne=7,specificSort=7

#%% Panel F + G
_,_,figs = getResponsivnessStats(r)
figs[0].get_axes()[0].get_images()[0].set_clim(0,440)
figs[-1].get_axes()[0].set_xlim(0,7)
figs[-1].get_axes()[0].set_ylim(0,1700)


