# -*- coding: utf-8 -*-
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
import statsmodels.formula.api as smf


# from FileManagement import *
# from GlmModule import *
from DataProcessing import *
from Visualisation import *
from GeneralRunners import *
from PCA import *
from StatTests import *
from Controls import *
from Behaviour import GetChoicePerAnimal


#%%Panels A, D

saveDir = "C:\\TaskProject\\2pData\\"
if not os.path.isdir(saveDir):
    os.makedirs(saveDir)
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["font.family"] = "Arial"

AllData, AllSubjects, AllSessions = LoadAllFiles("D:\\task_2p\\")
for i in range(17, 23):
    AllData[i]["planeDelays"] = AllData[i]["planeDelays"].T
sessId, nId = GetNeuronDetails(AllData)
tups = np.vstack((sessId, nId)).T
tups = list(tuple(map(tuple, tups)))

res, metadata = TestResponsiveness(AllData, 0.5, 0.5)
resp = res["Resp"]
p = res["pVals"]
z = res["zScores"]
r = p <= 0.05 / 8


plt.close("all")

sigs, sortInd = PrintPopulationResponsiveness(
    AllData, sortBy=r, addLines=False, returnIndex=True
)  #

getResponsivnessStats(r)


#%% Panels B, E
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

saveDir = "D:\\TaskProject\\EphysData\\"
if not os.path.isdir(saveDir):
    os.makedirs(saveDir)
plt.close("all")
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["font.family"] = "Arial"

AllData, AllSubjects, AllSessions = LoadAllFiles(
    "D:\\ProjectTask\\Ephys\\", ephys=True
)

sessId, pId, nId = GetNeuronDetails(AllData, ephys=True)

res, m = TestResponsivenessEphys(AllData, 0.5, 0.5)
resp = res["Resp"]
p = res["pVals"]
z = res["zScores"]
r = p <= 0.05 / 8

dpa = {
    "SS087": -298,
    "SS088": -330,
    "SS089": -338,
    "SS091": -314,
    "SS092": -310,
    "SS093": -277,
}

animals = np.vstack(AllSubjects.values())
super_border = [dpa[x] for x in np.squeeze(animals)]

d = [0, 1000]
sigs = PrintPopulationResponsivenessEphys(
    AllData,
    sortBy=sortby,
    addLines=False,
    w=40,
    depth=d,
    depthList=super_border,
)
depths = sigs[-1]
indDepths = np.where((depths >= d[0]) & (depths <= d[1]))[0]
rDepth = r[:, indDepths]
getResponsivnessStats(rDepth)


#%% Panels C, F
saveDir = "D:\\TaskProject\\EphysData\\"
if not os.path.isdir(saveDir):
    os.makedirs(saveDir)
plt.close("all")
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["font.family"] = "Arial"

AllData, AllSubjects, AllSessions = LoadAllFiles(
    "D:\\ProjectTask\\Ephys\\", ephys=True
)

sessId, pId, nId = GetNeuronDetails(AllData, ephys=True)

res, m = TestResponsivenessEphys(AllData, 0.5, 0.5)
resp = res["Resp"]
p = res["pVals"]
z = res["zScores"]
r = p <= 0.05 / 8

dpa = {
    "SS087": -298,
    "SS088": -330,
    "SS089": -338,
    "SS091": -314,
    "SS092": -310,
    "SS093": -277,
}

animals = np.vstack(AllSubjects.values())
super_border = [dpa[x] for x in np.squeeze(animals)]

d = [-1000, -0.00001]
sigs = PrintPopulationResponsivenessEphys(
    AllData,
    sortBy=sortby,
    addLines=False,
    w=40,
    depth=d,
    depthList=super_border,
)
depths = sigs[-1]
indDepths = np.where((depths >= d[0]) & (depths <= d[1]))[0]
rDepth = r[:, indDepths]
getResponsivnessStats(rDepth)
