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
from matplotlib.patches import Circle
from matplotlib.colors import ListedColormap
from matplotlib import cm

# from FileManagement import *
# from GlmModule import *
from DataProcessing import *
from Visualisation import *
from GeneralRunners import *
from PCA import *
from StatTests import *
from Controls import *
from Behaviour import GetChoicePerAnimal, GetReactionTime

saveDir = "C:\\TaskProject\\2pData\\"
if not os.path.isdir(saveDir):
    os.makedirs(saveDir)
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["font.family"] = "Arial"

# Load Data

AllData, AllSubjects, AllSessions = LoadAllFiles("D:\\task_2p\\")
for i in range(17, 23):
    AllData[i]["planeDelays"] = AllData[i]["planeDelays"].T
sessId, nId = GetNeuronDetails(AllData)
tups = np.vstack((sessId, nId)).T
tups = list(tuple(map(tuple, tups)))

with open(saveDir + "responsiveness.pickle", "rb") as f:
    res = pickle.load(f)
resp = res["Resp"]
p = res["pVals"]
z = res["zScores"]
r = p <= 0.05 / 8

modelsBase = GetOneFunctionVarianceDifferentModels(
    dataDir="D:\\Figures\\OneFunction_raw\\",
    cutoff=0.00001,
    shuffles=True,
    resp=tups_,
)

#%% Panels A-D
plt.close("all")

reps = 10

N = 16  # len(AllData)
stateText = ["rand", "pupil", "go", "correct", "reward"]
predictionImprov = np.zeros((5, N)) * np.nan
dfs = []
i = 0
ms = []
control = None

shuffleNumber = 100
shuffleDist = np.zeros((4, shuffleNumber))

ress = []

# States = 2: pupil, 3: Action, 4: outcome, 5: reward

for p in [0, 2, 3, 4, 5]:
    results, m = PredictionPerformanceSpecificPopulation(
        testStat=p,
        reps=reps,
        split=0.5,
        controlStat=control,
        C=5,
        controlState=0,
        preSplit=(False),
    )

    results = results[
        [0, 1, 2, 3, 4, 5, 6, 7, 11, 15, 16, 17, 18, 19, 20, 21], :, :
    ]
    ms.append(m)
    # results/=np.tile(np.max(np.max(results,2),1),(results.shape[-1],results.shape[-2],1)).T #normalised to highest
    Id = np.tile(np.arange(N), (reps, 2, 1)).T
    Reward = np.tile(np.tile([0, 1], (N, 1)).T, (reps, 1, 1)).T
    # # NeuronNum = np.tile(np.tile([1,5,10,25,50,100,1000],(500,1)).T,(16,2,1,1))
    RepsNum = np.tile(np.arange(reps), (N, 2, 1))
    dfs.append(
        pd.DataFrame(
            {
                "Animal": Id.flatten(),
                "Reward": Reward.flatten(),
                "RepNum": RepsNum.flatten(),
                "score": results.flatten(),
            }
        )
    )
    f, ax = plt.subplots(1)
    Id = np.tile(np.arange(N), (2, 1)).T
    Reward = np.tile(np.tile([0, 1], (N, 1)).T, (1, 1)).T
    d_tmp = pd.DataFrame(
        {
            "Animal": Id.flatten(),
            "Reward": Reward.flatten(),
            "score": np.nanmean(results, -1).flatten(),
        }
    )

    ax.scatter(
        np.nanmean(results, -1)[:, 0],
        np.nanmean(results, -1)[:, 1],
        c="k",
        edgecolors="black",
    )
    r = sp.stats.ttest_rel(
        np.nanmean(results, -1)[:, 0],
        np.nanmean(results, -1)[:, 1],
        nan_policy="omit",
    )
    ress.append(r)
    ax.plot(np.arange(-1, 1, 0.1), np.arange(-1, 1, 0.1), "k--")
    ax.set_aspect(1.0 / ax.get_data_ratio(), adjustable="box")
    f.set_size_inches(8, 8)
    ax.set_ylim(-0.2, 1)
    ax.set_xlim(-0.2, 1)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    # res = results[:,1,:]-results[:,0,:]
    res = np.nanmean(results, -1)
    res = res[:, 1] - res[:, 0]
    predictionImprov[i, :] = res
    i += 1


allDataSets = pd.concat(dfs[1:])

allTexts = []
for i in range(1, len(stateText)):
    allTexts.append(np.tile(stateText[i], (len(dfs[i]), 1)))
alltext = np.vstack(allTexts)
allDataSets["State"] = alltext

#%% Panel E

plt.close("all")

reps = 10

N = 16  # len(AllData)
stateText = ["rand", "pupil", "go", "correct", "reward"]
predictionImprov = np.zeros((5, N)) * np.nan
dfs = []
i = 0
ms = []
control = 2

shuffleNumber = 100
shuffleDist = np.zeros((4, shuffleNumber))

ress = []

# States = 2: pupil, 3: Action, 4: outcome, 5: reward

for p in [2, 3, 4]:
    for cs in [0, 1]:
        results, m = PredictionPerformanceSpecificPopulation(
            testStat=5,
            reps=reps,
            split=0.5,
            controlStat=p,
            C=5,
            controlState=cs,
            preSplit=(False),
        )

        results = results[
            [0, 1, 2, 3, 4, 5, 6, 7, 11, 15, 16, 17, 18, 19, 20, 21], :, :
        ]
        ms.append(m)
        # results/=np.tile(np.max(np.max(results,2),1),(results.shape[-1],results.shape[-2],1)).T #normalised to highest
        Id = np.tile(np.arange(N), (reps, 2, 1)).T
        Reward = np.tile(np.tile([0, 1], (N, 1)).T, (reps, 1, 1)).T
        # # NeuronNum = np.tile(np.tile([1,5,10,25,50,100,1000],(500,1)).T,(16,2,1,1))
        RepsNum = np.tile(np.arange(reps), (N, 2, 1))
        dfs.append(
            pd.DataFrame(
                {
                    "Animal": Id.flatten(),
                    "Reward": Reward.flatten(),
                    "RepNum": RepsNum.flatten(),
                    "score": results.flatten(),
                }
            )
        )
        f, ax = plt.subplots(1)
        Id = np.tile(np.arange(N), (2, 1)).T
        Reward = np.tile(np.tile([0, 1], (N, 1)).T, (1, 1)).T
        d_tmp = pd.DataFrame(
            {
                "Animal": Id.flatten(),
                "Reward": Reward.flatten(),
                "score": np.nanmean(results, -1).flatten(),
            }
        )

        ax.scatter(
            np.nanmean(results, -1)[:, 0],
            np.nanmean(results, -1)[:, 1],
            c="k",
            edgecolors="black",
        )

        ress.append(r)
        ax.plot(np.arange(-1, 1, 0.1), np.arange(-1, 1, 0.1), "k--")
        ax.set_aspect(1.0 / ax.get_data_ratio(), adjustable="box")
        f.set_size_inches(8, 8)
        ax.set_ylim(-0.2, 1)
        ax.set_xlim(-0.2, 1)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        # res = results[:,1,:]-results[:,0,:]
        res = np.nanmean(results, -1)
        res = res[:, 1] - res[:, 0]
        # predictionImprov[i, :] = res
        i += 1

#%% Panel F + G
s, c, p, shuff = PredictCorrectness(AllData[:16], opt=1)
plt.close("all")
shuffScore = np.nanpercentile(shuff, 95)
so = s[s > shuffScore]
su = s[s <= shuffScore]

cvalid = c[s > shuffScore, :]
pvalid = p[s > shuffScore, :]

f, ax = plt.subplots(1)
ax.scatter(np.zeros_like(so), so, c="k", s=60)
ax.scatter(np.zeros_like(su), su, c="grey", s=60)
ax.hlines(shuffScore, -0.5, 0.5, "k")
ax.set_xlim(-2, 2)

# ax.set_ylim(-0.1,0.45)
ax.set_ylabel("MCC")

ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
f.set_size_inches(8, 8)

pabs = pvalid.copy()
pabs[pvalid < 0.05] = 1
pabs[pvalid >= 0.05] = 0
coefTitles = np.tile(
    [
        "Baseline",
        "Zero",
        "L",
        "R",
        "Feedback N-1",
        "Feedback N-2",
        "Feedback N-3",
    ],
    (pvalid.shape[0], 1),
)
[0, 0.125, 0.25, 0.5, 0.75, 1]

df = pd.DataFrame(
    {
        "coefs": cvalid.flatten(),
        "param": coefTitles.flatten(),
        "pval": pabs.flatten(),
    }
)

f1, ax1 = plt.subplots(1)
ax1.hlines(0, -100, 100, "k", linestyles="dashed")
sns.stripplot(
    data=df,
    x="param",
    y="coefs",
    hue="pval",
    ax=ax1,
    palette=["grey", "black"],
    s=10,
)
sns.boxplot(
    data=df,
    x="param",
    y="coefs",
    ax=ax1,
    palette=["black"],
    medianprops={"visible": False},
    whiskerprops={"visible": False},
    showfliers=False,
    showbox=False,
    showcaps=False,
    showmeans=True,
    meanline=True,
    meanprops={"color": (0.4, 0.4, 0.4), "ls": "-", "lw": 2},
)

ax1.spines["right"].set_visible(False)
ax1.spines["top"].set_visible(False)
f1.set_size_inches(8, 8)
