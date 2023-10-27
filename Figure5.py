# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 10:46:20 2023

@author: LABadmin
"""
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
#%% Panel B
tups = np.vstack((sessId, nId)).T
tups = tups[r[0, :] == 1, :]
prog, progCon, pup, fb = PrintResponseProgression(AllData, fitTime=0.5)
ids = np.vstack(np.sort(np.array(modelsBase["Id"])))

maxLen = 0
for i in range(len(prog)):
    if prog[i].shape[0] > maxLen:
        maxLen = prog[i].shape[0]

progress = np.zeros((ids.shape[0], maxLen)) * np.nan

for iid in range(ids.shape[0]):
    sess = ids[iid, 0] - 1
    n = ids[iid, 1]
    p = prog[sess][:, n]
    progress[iid, : len(p)] = p


def CreateCI(ac, last=50, reps=500):
    ci = np.zeros((ac.shape[0], 2))
    sig = np.random.choice(ac[:], reps)
    for t in range(ac.shape[0]):
        # sig = np.roll(sig,1)

        ci[t, 0] = np.percentile(sig, 2.5)
        ci[t, 1] = np.percentile(sig, 97.5)

    return ci


plt.close("all")

acsTauFitTotal = []
exceedCi = []

iid = 7
prog1 = progress[np.where(ids[:, 0] == iid)[0], 0 : len(pup[iid - 1])]
acs = np.zeros((prog1.shape[0], prog1.shape[1] * 2 - 1))
acsTauFit = np.zeros((prog1.shape[0], 1))
corRange = np.array(range(-prog1.shape[1] + 1, prog1.shape[1]))

for pp in range(prog1.shape[0]):
    acs[pp, :] = sp.signal.correlate(prog1[pp, :], prog1[pp, :])
    af, ci = acf(prog1[pp, :], alpha=0.05, nlags=prog1[pp, :].shape[0])
    ci = CreateCI(af)
    Inds = np.where((af > ci[:, 1]))[0]
    Inds = Inds[np.where(np.diff(Inds, prepend=True) <= 1)]
    firstStop = np.where(np.diff(Inds, prepend=True) > 1)[0]
    if len(firstStop) > 0:
        Inds = Inds[: firstStop[0]]

    exceedCi.append(Inds)
    fitPart = acs[pp, corRange >= 0]

f, ax = plt.subplots(1, 1, sharex=True)
acsTauFitTotal.append(np.vstack(results))
f.suptitle("Session " + str(iid))
ax.plot(range(-prog1.shape[1] + 1, prog1.shape[1]), np.nanmean(acs, 0))
ci = CreateCI(np.nanmean(acs, 0).T)
plt.fill_between(
    range(-prog1.shape[1] + 1, prog1.shape[1]),
    ci[:, 0],
    ci[:, 1],
    alpha=0.2,
    facecolor="grey",
)
# ax[1].set_xlim(-20,20)
ax.set_title("Average Trace")
ax.set_xlabel("lag (trials)")
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
#%% Panel C
lastInd = []
exceedCi = []

tups = np.vstack((sessId, nId)).T
tups = tups[r[0, :] == 1, :]
ids = np.vstack(np.sort(np.array(modelsBase["Id"])))


for iid in np.unique(ids[:, 0]):  # range(len(np.unique(ids[:,0]))):
    prog1 = progress[np.where(ids[:, 0] == iid)[0], 0 : len(pup[iid - 1])]
    acs = np.zeros((prog1.shape[0], prog1.shape[1] * 2 - 1))
    acsTauFit = np.zeros((prog1.shape[0], 1))
    corRange = np.array(range(-prog1.shape[1] + 1, prog1.shape[1]))

    for pp in range(prog1.shape[0]):
        acs[pp, :] = sp.signal.correlate(prog1[pp, :], prog1[pp, :])
        af, ci = acf(prog1[pp, :], alpha=0.05, nlags=prog1[pp, :].shape[0])
        # ci =CreateCIShift(prog1[pp,:])
        ci = CreateCI(af)
        # Find where correlation exceeds 95% far confidence interval
        Inds = np.where((af > ci[:, 1]))[0]
        Inds = Inds[np.where(np.diff(Inds, prepend=True) <= 1)]
        firstStop = np.where(np.diff(Inds, prepend=True) > 1)[0]
        Inds = np.where((af > ci[:, 1]))[0]
        Inds = Inds[np.where(np.diff(Inds, prepend=True) <= 1)]
        firstStop = np.where(np.diff(Inds, prepend=True) > 1)[0]
        if len(firstStop) > 0:
            Inds = Inds[: firstStop[0]]
        # Inds = Inds[Inds<50]
        exceedCi.append(Inds)

last = []
for i in range(len(exceedCi)):
    if len(exceedCi[i]) > 0:
        last.append(exceedCi[i][-1])
    else:
        last.append(0)
sns.histplot(last, bins=9)
plt.xlim(0, 4)

#%% Panel D + E
plt.close("all")
tups = np.vstack((sessId, nId)).T
tups = tups[r[0, :] == 1, :]

modelsBase = GetOneFunctionVarianceDifferentModels(
    dataDir="D:\\Figures\\OneFunction_raw\\",
    cutoff=0.01,
    shuffles=True,
    resp=tups,
).sort_index()

modelsLick = GetOneFunctionVarianceDifferentModels(
    dataDir="D:\\Figures\\OneFunction_raw_NoStimLick\\",
    cutoff=0.01,
    shuffles=True,
    resp=tups,
).sort_index()

# run multiple times for different conditions (1 - was sig then not, 2- was not sig but now is, 3 - stayed sig)

f, ax = plt.subplots(1, 2)

colors = [(0.8, 0.8, 0.8), (0.5, 0.5, 0.5), (0, 0, 0)]
sF = []
sP = []
for cond in [1, 2, 3]:
    if cond == 3:
        lickSharedFi = pd.Series(
            list(
                set(modelsBase[modelsBase.bestFit == "F"]["Id"]).intersection(
                    set(modelsLick[modelsLick.bestFit == "F"]["Id"])
                )
            )
        ).sort_index()
        lickSharedPFi = list(
            set(modelsBase[modelsBase.bestFit == "PF"]["Id"]).intersection(
                set(modelsLick[modelsLick.bestFit == "PF"]["Id"])
            )
        )
        lickSharedPi = list(
            set(modelsBase[modelsBase.bestFit == "P"]["Id"]).intersection(
                set(modelsLick[modelsLick.bestFit == "P"]["Id"])
            )
        )
    if cond == 2:
        lickSharedFi = pd.Series(
            list(
                set(modelsBase[modelsBase.bestFit != "F"]["Id"]).intersection(
                    set(modelsLick[modelsLick.bestFit == "F"]["Id"])
                )
            )
        ).sort_index()
        lickSharedPFi = list(
            set(modelsBase[modelsBase.bestFit != "PF"]["Id"]).intersection(
                set(modelsLick[modelsLick.bestFit == "PF"]["Id"])
            )
        )
        lickSharedPi = list(
            set(modelsBase[modelsBase.bestFit != "P"]["Id"]).intersection(
                set(modelsLick[modelsLick.bestFit == "P"]["Id"])
            )
        )
    if cond == 1:
        lickSharedFi = pd.Series(
            list(
                set(modelsBase[modelsBase.bestFit == "F"]["Id"]).intersection(
                    set(modelsLick[modelsLick.bestFit != "F"]["Id"])
                )
            )
        ).sort_index()
        lickSharedPFi = list(
            set(modelsBase[modelsBase.bestFit == "PF"]["Id"]).intersection(
                set(modelsLick[modelsLick.bestFit != "PF"]["Id"])
            )
        )
        lickSharedPi = list(
            set(modelsBase[modelsBase.bestFit == "P"]["Id"]).intersection(
                set(modelsLick[modelsLick.bestFit != "P"]["Id"])
            )
        )
    sharedF = []
    sharedP = []

    for share in lickSharedFi:
        fitBase = modelsBase[modelsBase.Id.reset_index(drop=True) == share][
            "fit"
        ].item()["fitInclude"][4, :]
        fitLick = modelsLick[modelsLick.Id.reset_index(drop=True) == share][
            "fit"
        ].item()["fitInclude"][4, :]
        fitBase = fitBase[4] / (0.5 * (2 * fitBase[0] + fitBase[4]))
        fitLick = fitLick[4] / (0.5 * (2 * fitLick[0] + fitLick[4]))
        sharedF.append((fitBase, fitLick))

    for share in lickSharedPi:
        fitBase = modelsBase[modelsBase.Id.reset_index(drop=True) == share][
            "fit"
        ].item()["fitInclude"][1, :]
        fitLick = modelsLick[modelsLick.Id.reset_index(drop=True) == share][
            "fit"
        ].item()["fitInclude"][1, :]
        fitBase = fitBase[1] / (0.5 * (2 * fitBase[0] + fitBase[1]))
        fitLick = fitLick[1] / (0.5 * (2 * fitLick[0] + fitLick[1]))
        sharedP.append((fitBase, fitLick))

    sharedF = np.vstack(sharedF).T
    sharedP = np.vstack(sharedP).T

    sF.append(sharedF)
    sP.append(sharedP)

    ax[0].scatter(*sharedF, s=30, color=colors[cond - 1])
    ax[0].set_aspect("equal", adjustable="box")
    ax[0].set_xlim(-2, 2)
    ax[0].set_ylim(-2, 2)
    ax[0].plot(np.arange(-2, 2, 0.1), np.arange(-2, 2, 0.1), "k--")
    ax[0].set_title("R (gain) feedback")
    ax[0].set_xlabel("Baseline R gain")
    ax[0].set_ylabel("No Lick Baseline R gain")
    ax[0].spines["right"].set_visible(False)
    ax[0].spines["top"].set_visible(False)
    f.set_size_inches(8, 8)

    ax[1].scatter(*sharedP, s=30, color=colors[cond - 1])
    ax[1].set_aspect("equal", adjustable="box")
    ax[1].set_xlim(-2, 2)
    ax[1].set_ylim(-2, 2)
    ax[1].plot(np.arange(-2, 2, 0.1), np.arange(-2, 2, 0.1), "k--")
    ax[1].set_title("R (gain) Pupil")
    ax[1].set_xlabel("Baseline R gain")
    ax[1].set_ylabel("No Lick Baseline R gain")
    ax[1].spines["right"].set_visible(False)
    ax[1].spines["top"].set_visible(False)
    f.set_size_inches(8, 8)
f, axx = plt.subplots(1)
axx.bar([1, 2, 3], [1, 1, 1], color=colors)
axx.set_title("1 - sig/not 2 - not/sig 3 - sig/sig")

#%% G-H
results = []
cumFbs = []

for i in range(1, len(AllData) + 1):
    feedback = AllData[i]["feedbackType"][:, 0]
    if AllData[i]["ImagingSide"] == 1:
        contra = AllData[i]["contrastLeft"][:, 0]
        ipsi = AllData[i]["contrastRight"][:, 0]

    else:
        ipsi = AllData[i]["contrastLeft"][:, 0]
        contra = AllData[i]["contrastRight"][:, 0]

    feedbackContra = (feedback == 1) & (contra > 0)
    feedbackContra = feedbackContra.astype(int)
    feedbackContra[(feedback == 0) & (contra > 0)] = -1
    feedbackIpsi = (feedback == 1) & (ipsi > 0)

    I = i
    if AllData[I]["ImagingSide"] == 1:
        contra = AllData[I]["contrastLeft"][:, 0]
        ipsi = AllData[I]["contrastRight"][:, 0]

    else:
        ipsi = AllData[I]["contrastLeft"][:, 0]
        contra = AllData[I]["contrastRight"][:, 0]

    wh, t = AlignStim(
        AllData[I]["wheelVelocity"],
        AllData[I]["wheelTimes"],
        AllData[I]["stimT"][:, 0],
        np.array([-0.5, 0.5]).reshape(1, -1),
    )
    wh = wh[((t <= 0.5) & (t >= 0)), :, 0]
    wh = np.nanmean(np.abs(wh), 0)
    moveDuringStimInd = np.where(wh > 0)[0]

    delays = AllData[I]["planeDelays"]
    planes = AllData[I]["planes"]
    sig = AllData[I]["calTrace"]
    times = AllData[I]["calTimes"]

    ca_raw, t = GetCalciumAligned(
        sig,
        times,
        AllData[I]["stimT"][:, 0],
        np.array([-0.5, 0.5]).reshape(1, -1),
        planes,
        delays,
    )
    ca_raw = ca_raw / np.nanmax(np.nanmax(ca_raw, 1), 0)
    ca = ca_raw - np.tile(
        np.nanmean(ca_raw[t <= 0, :, :], axis=0), (ca_raw.shape[0], 1, 1)
    )
    testTime = np.where((t > 0) & (t < 0.5))[0]

    avgs = np.nanmean(ca[testTime, :, :], 0)
    avgsRaw = np.nanmean(ca_raw[testTime, :, :], 0)

    delInd = np.where(np.isnan(avgs[:, 0]))[0]

    delInd = np.union1d(delInd, moveDuringStimInd)

    # Measure cumulative feedback
    # splitFb = np.split(
    #     feedbackContra.astype(int), np.where(feedbackContra == 0)[0]
    # )
    # cumFeedbackContra = np.zeros(len(feedbackContra))
    # count = 0
    # for p in range(len(splitFb)):
    #     part = splitFb[p]
    #     if len(part) == 0:
    #         cumFeedbackContra[count] = 0
    #         count = 0
    #         continue
    #     # part = np.invert(part)
    #     part = np.cumsum(part)
    #     cumFeedbackContra[count : count + len(part)] = part
    #     count = count + len(part + 1)

    # feedbackContraRev = np.invert(feedbackContra)
    # splitFb = np.split(feedbackContra, np.where((feedbackContraRev) == -1)[0])
    # cumNoneContra = np.zeros(len(feedbackContraRev))
    # count = 0
    # for p in range(len(splitFb)):
    #     part = splitFb[p]
    #     if len(part) == 0:
    #         cumNoneContra[count] = 0
    #         count = 0
    #         continue
    #     # part = np.invert(part)
    #     part = np.cumsum(part)
    #     cumNoneContra[count : count + len(part)] = part
    #     count = count + len(part + 1)
    # # measure average number of accumulations
    # pospos = []
    # negpos = []
    # posneg = []
    # negneg = []

    # cumFeedbackContra = np.cumsum(feedbackContra)
    cumFeedbackContra = np.zeros_like(feedbackContra)
    cumFeedbackContra[0] = 0
    for index in range(1, len(feedbackContra)):
        nextSum = cumFeedbackContra[index - 1] + feedbackContra[index]
        if nextSum < 0:
            nextSum = 0
        cumFeedbackContra[index] = nextSum

    prevFB = np.append([0], feedback[1:])

    cumFbs.append(cumFeedbackContra)
    strats = np.zeros_like(prevFB)
    strats[: len(strats) // 2] = 1
    X_train, X_test, Y_train, Y_test = train_test_split(
        prevFB,
        np.arange(len(prevFB)),
        test_size=0.80,
        stratify=strats,
    )

    for j in range(len(feedbackContra)):
        if j in delInd:
            continue
        # if j in Y_test:
        #     continue
        if (j - 1) < 0:
            results.append(
                {
                    "Session": i,
                    "Subject": AllSubjects[i],
                    "cumFb": cumFeedbackContra[j],
                    "prev": 0,
                    "contrast": contra[j],
                }
            )
        results.append(
            {
                "Session": i,
                "Subject": AllSubjects[i],
                "cumFb": cumFeedbackContra[j],
                "prev": feedback[j - 1],
                "contrast": contra[j],
            }
        )
        # if (feedbackContra[j]==1):
        # pospos.append(cumFeedbackContra[j])
        # negpos.append(cumNoneContra[j])

        # else:
        # posneg.append(cumFeedbackContra[j])
        # negneg.append(cumNoneContra[j])
results = pd.DataFrame(results)

plt.close("all")
f, ax = plt.subplots(1, 1, sharex=True)
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)

sns.stripplot(
    data=results[(results.contrast > 0) & (results.Session == 16)],
    x="prev",
    y="cumFb",
    ax=ax,
)

f, ax = plt.subplots(1, 1, sharex=True)
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)

sns.stripplot(
    data=results[(results.contrast > 0) & (results.Session == 9)],
    x="prev",
    y="cumFb",
    ax=ax,
)

a = (
    results[(results.contrast >= 0) & (results.prev == 1)]
    .groupby("Session")
    .median()
    .cumFb
).to_numpy()
b = (
    results[(results.contrast >= 0) & (results.prev == 0)]
    .groupby("Session")
    .median()
    .cumFb
).to_numpy()

f, ax = plt.subplots(1)
ax.scatter(a, b, facecolor="black")
ax.plot(np.arange(-1, 60, 0.5), np.arange(-1, 60, 0.5), "k--")
ax.set_xlim(0, 55)
ax.set_ylim(0, 55)
ax.set_xlabel("Previously Positive Feedback")
ax.set_ylabel("Previously Negative Feedback")
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.set_aspect("equal", adjustable="box")

#%% Panel J
from sklearn.metrics import explained_variance_score


def contrastFunction(X, R0, ap, ag, ac, af, score, c50, n):
    R = (
        R0
        + ap * X[1, :]
        + ag * X[2, :]
        + ac * X[3, :]
        + af * X[4, :]
        + score * X[5, :]
    )
    c = X[0, :]
    # try to add constraint

    if (R0 + ap * 1 + ag * 1 + ac * 1 + af * 1 + score * 1) < 0:
        return -10000 * np.ones(len(c))
    if (R0 + ap) < 0:
        return -100000 * np.ones(len(c))
    if (R0 + ag * 1) < 0:
        return -100000 * np.ones(len(c))
    if (R0 + ac * 1) < 0:
        return -100000 * np.ones(len(c))
    if (R0 + af * 1) < 0:
        return -100000 * np.ones(len(c))
    if (R0 + score * 1) < 0:
        return -100000 * np.ones(len(c))
    return R * (c**n / (c50**n + c**n))


def contrastFunctionFirst(X, R0, ap, ag, ac, af, c50, n):
    R = R0 + ap * X[1, :] + ag * X[2, :] + ac * X[3, :] + af * X[4, :]
    c = X[0, :]
    # try to add constraint

    if (R0 + ap * 1 + ag * 1 + ac * 1 + af * 1) < 0:
        return -10000 * np.ones(len(c))
    if (R0 + ap) < 0:
        return -100000 * np.ones(len(c))
    if (R0 + ag * 1) < 0:
        return -100000 * np.ones(len(c))
    if (R0 + ac * 1) < 0:
        return -100000 * np.ones(len(c))
    if (R0 + af * 1) < 0:
        return -100000 * np.ones(len(c))
    return R * (c**n / (c50**n + c**n))


tups = np.vstack((sessId, nId)).T
tups = tups[r[0, :] == 1, :]
modelsBase = GetOneFunctionVarianceDifferentModels(
    dataDir="D:\\Figures\\OneFunction_raw\\",
    cutoff=0.01,
    shuffles=True,
    resp=tups,
)

fits = modelsBase["fit"].to_numpy()
Ids = np.vstack(modelsBase["Id"].to_numpy())[:, 0]

propsList = []
propsList2 = []
variance = []
for i in range(len(fits)):

    X = fits[i]["X"]
    X = np.vstack((X, np.append([0], X[-1, :-1])))
    Y = fits[i]["Y"]

    I = Ids[i]
    if AllData[I]["ImagingSide"] == 1:
        contra = AllData[I]["contrastLeft"][:, 0]
        ipsi = AllData[I]["contrastRight"][:, 0]

    else:
        ipsi = AllData[I]["contrastLeft"][:, 0]
        contra = AllData[I]["contrastRight"][:, 0]

    wh, t = AlignStim(
        AllData[I]["wheelVelocity"],
        AllData[I]["wheelTimes"],
        AllData[I]["stimT"][:, 0],
        np.array([-0.5, 0.5]).reshape(1, -1),
    )
    wh = wh[((t <= 0.5) & (t >= 0)), :, 0]
    wh = np.nanmean(np.abs(wh), 0)
    moveDuringStimInd = np.where(wh > 0)[0]

    delays = AllData[I]["planeDelays"]
    planes = AllData[I]["planes"]
    sig = AllData[I]["calTrace"]
    times = AllData[I]["calTimes"]

    feedback = AllData[I]["feedbackType"][:, 0]
    feedbackContra = (feedback == 1) & (contra > 0)
    feedbackContra = feedbackContra.astype(int)
    feedbackContra[(feedback == 0) & (contra > 0)] = -1
    feedbackIpsi = (feedback == 1) & (ipsi > 0)

    cumFeedbackContra = np.zeros_like(feedbackContra)
    cumFeedbackContra[0] = 0
    for index in range(1, len(feedbackContra)):
        nextSum = cumFeedbackContra[index - 1] + feedbackContra[index]
        if nextSum < 0:
            nextSum = 0
        cumFeedbackContra[index] = nextSum

    ca_raw, t = GetCalciumAligned(
        sig,
        times,
        AllData[I]["stimT"][:, 0],
        np.array([-0.5, 0.5]).reshape(1, -1),
        planes,
        delays,
    )
    ca_raw = ca_raw / np.nanmax(np.nanmax(ca_raw, 1), 0)
    ca = ca_raw - np.tile(
        np.nanmean(ca_raw[t <= 0, :, :], axis=0), (ca_raw.shape[0], 1, 1)
    )
    testTime = np.where((t > 0) & (t < 0.5))[0]

    avgs = np.nanmean(ca[testTime, :, :], 0)
    avgsRaw = np.nanmean(ca_raw[testTime, :, :], 0)

    delInd = np.where(np.isnan(avgs[:, 0]))[0]

    delInd = np.union1d(delInd, moveDuringStimInd)

    contra = np.delete(contra, delInd)
    ipsi = np.delete(ipsi, delInd)
    cumFeedbackContra = np.delete(cumFeedbackContra, delInd)

    prevRewardContra = (X[-2, :] == 1) & (contra == 1)
    prevRewardIpsi = (X[-2, :] == 1) & (ipsi == 1)
    prevRewardNothing = (X[-2, :] == 1) & (ipsi == 0) & (contra == 0)
    prevRewardContra = np.append([0], prevRewardContra[1:])
    prevRewardIpsi = np.append([0], prevRewardIpsi[1:])
    prevRewardNothing = np.append([0], prevRewardNothing[1:])

    X = np.vstack((X, cumFeedbackContra))

    # p0 = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5,0.5, 2]
    # bounds = ([0, -1, -1, -1, -1,-1, 0.1, 1], [1, 1, 1, 1, 1, 1,1, 5])
    p0 = [0.5, 0, 0, 0, 0, 0, 0.5, 2]
    bounds = (
        [
            0,
            -np.finfo(float).eps,
            -np.finfo(float).eps,
            -np.finfo(float).eps,
            -1,
            -1,
            0.1,
            1,
        ],
        [
            1,
            np.finfo(float).eps,
            np.finfo(float).eps,
            np.finfo(float).eps,
            1,
            1,
            1,
            5,
        ],
    )
    try:
        X_train, X_test, Y_train, Y_test = train_test_split(
            X.T, Y, test_size=0.20
        )

        X_train = X_train.T
        X_test = X_test.T

        props_, _ = sp.optimize.curve_fit(
            contrastFunction, X_train, Y_train, p0=p0, bounds=bounds
        )
        propsList.append(props_)

        v1 = explained_variance_score(
            Y_test, contrastFunction(X_test, *props_)
        )

    except:
        print(traceback.format_exc())
        propsList.append(np.zeros(8) * np.nan)
        v1 = 0
    try:

        X_train, X_test, Y_train, Y_test = train_test_split(
            X.T, Y, test_size=0.20
        )

        X_train = X_train.T
        X_test = X_test.T
        p0 = [0.5, 0, 0, 0, 0, 0.5, 2]
        bounds = (
            [
                0,
                -np.finfo(float).eps,
                -np.finfo(float).eps,
                -np.finfo(float).eps,
                -1,
                0.1,
                1,
            ],
            [
                1,
                np.finfo(float).eps,
                np.finfo(float).eps,
                np.finfo(float).eps,
                1,
                1,
                5,
            ],
        )

        props_, _ = sp.optimize.curve_fit(
            contrastFunctionFirst, X_train, Y_train, p0=p0, bounds=bounds
        )
        propsList2.append(props_)

        v2 = explained_variance_score(
            Y_test, contrastFunctionFirst(X_test, *props_)
        )
        # variance.append((v2, v1))
    except:
        print("second fit")
        print(traceback.format_exc())
        propsList.append(np.zeros(8) * np.nan)
        v2 = 0
    variance.append((v2, v1))


feedbackInd = np.where(
    (modelsBase["bestFit"] == "F") | (modelsBase["bestFit"] == "PF")
)[0]

nofeedbackInd = np.where(
    (modelsBase["bestFit"] != "F") & (modelsBase["bestFit"] != "PF")
)[0]

f1 = modelsBase["fit"][feedbackInd]
fr = GetRFromFitList(list(f1), "F")
f, ax = plt.subplots(1)
frScore = np.vstack(propsList)[feedbackInd, :]
frScore = frScore[:, [0, 4]]

frm = (fr[:, 0] + fr[:, 1] - fr[:, 0]) / (
    0.5 * (fr[:, 0] + fr[:, 1] + fr[:, 0])
)

fsrm = (frScore[:, 0] + frScore[:, 1] - frScore[:, 0]) / (
    0.5 * (frScore[:, 0] + frScore[:, 1] + frScore[:, 0])
)

f1n = modelsBase["fit"][nofeedbackInd]
frn = GetRFromFitList(list(f1n), "F")

frScoren = np.vstack(propsList)[nofeedbackInd, :]
frScoren = frScoren[:, [0, 4]]

frmn = (frn[:, 0] + frn[:, 1] - frn[:, 0]) / (
    0.5 * (frn[:, 0] + frn[:, 1] + frn[:, 0])
)

fsrmn = (frScoren[:, 0] + frScoren[:, 1] - frScoren[:, 0]) / (
    0.5 * (frScoren[:, 0] + frScoren[:, 1] + frScoren[:, 0])
)
f, ax = plt.subplots(1)
ax.scatter(frmn, fsrmn, facecolor="grey")
ax.scatter(frm, fsrm, facecolor="black")
ax.plot(np.arange(-2, 2, 0.5), np.arange(-2, 2, 0.5), "k--")
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_xlabel("RM control")
ax.set_ylabel("RM with stimulus score")
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.set_aspect("equal", adjustable="box")
#%% Panel K
def contrastFunction(X, R0, ap, ag, ac, af, af2, c50, n):
    R = (
        R0
        + ap * X[1, :]
        + ag * X[2, :]
        + ac * X[3, :]
        + af * X[4, :]
        + af2 * X[5, :]
    )
    c = X[0, :]
    # try to add constraint
    if any(R < 0):
        return -100000 * np.ones(len(c))

    if (R0 + ap * 1 + ag * 1 + ac * 1 + af * 1) < 0:
        return -100000 * np.ones(len(c))
    if (R0 + ap) < 0:
        return -100000 * np.ones(len(c))
    if (R0 + ag * 1) < 0:
        return -100000 * np.ones(len(c))
    if (R0 + ac * 1) < 0:
        return -100000 * np.ones(len(c))
    if (R0 + af * 1) < 0:
        return -100000 * np.ones(len(c))
    if (R0 + af2 * 1) < 0:
        return -100000 * np.ones(len(c))
    return R * (c**n / (c50**n + c**n))


tups = np.vstack((sessId, nId)).T
tups = tups[r[0, :] == 1, :]
modelsBase = GetOneFunctionVarianceDifferentModels(
    dataDir="D:\\Figures\\OneFunction_raw\\",
    cutoff=0.01,
    shuffles=True,
    resp=tups,
)

fits = modelsBase["fit"].to_numpy()
propsList = []
for i in range(len(fits)):

    X = fits[i]["X"]
    X = np.vstack((X, np.append([0], X[-1, :-1])))
    Y = fits[i]["Y"]

    # p0 = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5,0.5, 2]
    # bounds = ([0, -1, -1, -1, -1,-1, 0.1, 1], [1, 1, 1, 1, 1, 1,1, 5])
    p0 = [0.5, 0, 0, 0, 0, 0, 0.5, 2]
    bounds = (
        [
            0,
            -np.finfo(float).eps,
            -np.finfo(float).eps,
            -np.finfo(float).eps,
            -1,
            -1,
            0.1,
            1,
        ],
        [
            1,
            np.finfo(float).eps,
            np.finfo(float).eps,
            np.finfo(float).eps,
            1,
            1,
            1,
            5,
        ],
    )
    try:
        props_, _ = sp.optimize.curve_fit(
            contrastFunction, X, Y, p0=p0, bounds=bounds
        )
        propsList.append(props_)
    except:
        print(traceback.format_exc())
        propsList.append(np.zeros(8) * np.nan)
#%% CNTD
f1 = np.array(modelsBase["fit"])
# fp1 = np.array(modelsBase[modelsBase["bestFit"] == "PF"]["fit"])
fr = GetRFromFitList(list(f1), "F")
# fpr = GetRFromFitList(list(fp1), "PF")

valsPos = (fr[:, 1]) / (0.5 * (2 * fr[:, 0] + fr[:, 1]))

feedbackInd = np.where(
    (modelsBase["bestFit"] == "F") | (modelsBase["bestFit"] == "PF")
)[0]
nofeedbackInd = np.where(
    (modelsBase["bestFit"] != "F") & (modelsBase["bestFit"] != "PF")
)[0]

# valsPos = np.append(valsPos,valPos2)

props = np.vstack(propsList)

valPosN1 = (props[:, 4]) / (
    0.5 * (2 * props[:, 0] + props[:, 4] + props[:, 5])
)
valPosN2 = (props[:, 5]) / (
    0.5 * (2 * props[:, 0] + props[:, 5] + props[:, 4])
)

f, ax = plt.subplots(1, 2, sharex=True, sharey=True)


ax[0].scatter(
    valsPos[nofeedbackInd], valPosN1[nofeedbackInd], facecolor="grey"
)
ax[0].scatter(valsPos[feedbackInd], valPosN1[feedbackInd], facecolor="black")
ax[0].plot(np.arange(-3, 3, 0.5), np.arange(-3, 3, 0.5), "k--")
ax[0].set_title("Normal Fit vs. New double fit")
ax[0].set_xlabel("Modulation Index (%) First N-1 Model")
ax[0].set_ylabel("Modulation Index (%) N-1 and N-2 Model")
ax[0].spines["right"].set_visible(False)
ax[0].spines["top"].set_visible(False)
ax[0].set_aspect("equal", adjustable="box")
ax[0].set_xlim(-2.2, 2.2)
ax[0].set_ylim(-2.2, 2.2)

ax[1].scatter(
    valPosN1[nofeedbackInd], valPosN2[nofeedbackInd], facecolor="grey"
)
ax[1].scatter(valPosN1[feedbackInd], valPosN2[feedbackInd], facecolor="black")
ax[1].plot(np.arange(-3, 3, 0.5), np.arange(-3, 3, 0.5), "k--")
ax[1].set_title("Modulation Index by N-1 and N-2")
ax[1].set_xlabel("Modulation Index (%) N-1 Feedback")
ax[1].set_ylabel("Modulation Index (%) N-2 Feedback")
ax[1].spines["right"].set_visible(False)
ax[1].spines["top"].set_visible(False)
ax[1].set_aspect("equal", adjustable="box")
