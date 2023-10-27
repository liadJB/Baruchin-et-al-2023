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

#%% Panel B
GetPupilSize(AllData, "feedback", plot=True)
#%% Panel C

f1 = np.array(modelsBase[modelsBase["bestFit"] == "F"]["fit"])
fp1 = np.array(modelsBase[modelsBase["bestFit"] == "PF"]["fit"])
p1 = np.array(modelsBase[modelsBase["bestFit"] == "P"]["fit"])
nf1 = np.array(modelsBase[modelsBase["bestFit"] != "F"]["fit"])
np1 = np.array(modelsBase[modelsBase["bestFit"] != "P"]["fit"])
nfp1 = np.array(modelsBase[modelsBase["bestFit"] != "PF"]["fit"])
g1 = np.array(modelsBase[modelsBase["bestFit"] == "G"]["fit"])
c1 = np.array(modelsBase[modelsBase["bestFit"] == "C"]["fit"])
ng1 = np.array(modelsBase[modelsBase["bestFit"] != "G"]["fit"])
nc1 = np.array(modelsBase[modelsBase["bestFit"] != "C"]["fit"])
gc1 = np.array(modelsBase[modelsBase["bestFit"] != "GC"]["fit"])
allFit = np.array(modelsBase["fit"])
grId = np.array(modelsBase[modelsBase["bestFit"] == "G"]["Id"])
ngrId = np.array(modelsBase[modelsBase["bestFit"] != "G"]["Id"])
grallId = np.array(modelsBase["Id"])

fr = GetRFromFitList(list(f1), "F")
fpr = GetRFromFitList(list(fp1), "PF")
nfr = GetRFromFitList(list(nf1), "F")
pr = GetRFromFitList(list(p1), "P")
npr = GetRFromFitList(list(np1), "P")
nfpr = GetRFromFitList(list(nfp1), "PF")
gr = GetRFromFitList(list(g1), "G")
ngr = GetRFromFitList(list(ng1), "G")
cr = GetRFromFitList(list(c1), "C")
ncr = GetRFromFitList(list(nc1), "C")

frall = GetRFromFitList(list(allFit), "F")
prall = GetRFromFitList(list(allFit), "P")
grall = GetRFromFitList(list(allFit), "G")
crall = GetRFromFitList(list(allFit), "C")
gr = gr[np.vstack(grId)[:, 0] <= 16]
ngr = ngr[np.vstack(ngrId)[:, 0] <= 16]
grall = grall[np.vstack(grallId)[:, 0] <= 16]


f, ax = plt.subplots(1)
###OPTIONAL
# This is the original measure
# ax.scatter(nfpr[:,1],nfpr[:,2],s=30,c=['w'],edgecolors='grey')
# ax.scatter(fpr[:,1],fpr[:,2],s=30,c=['k'],edgecolors='k')

# this is in modulation

# nfpr[np.sum(nfpr[:,[0,1]],1)<0,1] = -nfpr[np.sum(nfpr[:,[0,1]],1)<0,0]

# nfpr[np.sum(nfpr[:,[0,2]],1)<0,2] = -nfpr[np.sum(nfpr[:,[0,2]],1)<0,0]


# fpr[np.sum(fpr[:,[0,1]],1)<0,1] = -fpr[np.sum(fpr[:,[0,1]],1)<0,0]

# fpr[np.sum(fpr[:,[0,2]],1)<0,2] = -fpr[np.sum(fpr[:,[0,2]],1)<0,0]

# fpr[np.sum(fpr[:,[0,2]],1)<0,2] = -fpr[np.sum(fpr[:,[0,2]],1)<0,0]

# nfpr[np.sum(nfpr[:,[0,1]],1)<0,:] = np.nan
# nfpr[np.sum(nfpr[:,[0,2]],1)<0,:] = np.nan
# fpr[np.sum(fpr[:,[0,1]],1)<0,:] = np.nan
# fpr[np.sum(fpr[:,[0,2]],1)<0,:] = np.nan

a = (nfpr[:, 1]) / (0.5 * (2 * nfpr[:, 0] + nfpr[:, 1]))
b = (nfpr[:, 2]) / (0.5 * (2 * nfpr[:, 0] + nfpr[:, 2]))
ax.scatter(a, b, s=30, c=(0.6, 0.6, 0.6))
a = (fpr[:, 1]) / (0.5 * (2 * fpr[:, 0] + fpr[:, 1]))
b = (fpr[:, 2]) / (0.5 * (2 * fpr[:, 0] + fpr[:, 2]))
# a = np.delete(a,np.where(abs(b)>2)[0])
# b  = np.delete(b,np.where(abs(b)>2)[0])
ax.scatter(a, b, s=30, c=["k"])


ax.set_aspect("equal", adjustable="box")
# ax.set_xlim(-0.6,0.6)
# ax.set_ylim(-0.6,0.6)
# ax.set_xlim(-2,2)
# ax.set_ylim(-2,2)
# ax.plot(np.arange(-0.7,0.7,0.1),np.arange(-0.7,0.7,0.1),'k--')
# ax.plot(np.arange(-2,2,0.1),np.arange(-2,2,0.1),'k--')
ax.hlines(0, -2, 2, "k", linestyles="dashed")
ax.vlines(0, -2, 2, "k", linestyles="dashed")
ax.set_title("Pupil gain vs. Feeback gain")
ax.set_xlabel("Pupil gain")
ax.set_ylabel("Feedback gain")
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)

#%% Panel D

plt.close("all")
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from sklearn.impute import SimpleImputer


def get_pupil_pos(frameTimes, stimStarts, pos):
    xyPos = np.zeros((len(stimStarts), 2))
    for i, s in enumerate(stimStarts):
        xyPos[i, :] = pos[np.where((frameTimes - s) <= 0)[0][-1], :]
    return xyPos


def centroidDiff(X, classification: bool):
    a = np.sqrt(
        np.sum(
            (
                np.nanmean(X[classification == 1, :], 0)
                - np.nanmean(X[classification == 0, :], 0)
            )
            ** 2
        )
    )
    return a


summaries = []
slScores = []
pVals = []
tups = np.vstack((sessId, nId)).T
posList = []
negList = []
generalList = []
for n in range(1, len(AllData) + 1):
    d = AllData[n]

    stimStarts = d["stimT"][:, 0]
    fbType = d["feedbackType"]
    prevReward = np.append(0, fbType[:-1, 0])
    pupilXy = d["pupilxy"]
    times = d["pupilTimes"]

    sig = d["calTrace"]
    # normalise sig
    # sig/=np.nanmax(sig,axis = 0)
    delays = d["planeDelays"]
    planes = d["planes"]
    times = d["calTimes"]
    imagingSide = d["ImagingSide"]

    if imagingSide == 1:
        contra = d["contrastLeft"][:, 0]
        ipsi = d["contrastRight"][:, 0]

    else:
        ipsi = d["contrastLeft"][:, 0]
        contra = d["contrastRight"][:, 0]
    ch = d["choice"][:, 0]
    beepStarts = d["goCueTimes"]
    feedbackStarts = d["feedbackTimes"]
    feedBackPeriod = d["posFeedbackPeriod"]
    feedbackType = d["feedbackType"][:, 0]
    prevFeedback = np.append(0, feedbackType[:-1])

    pupilXy_trials = get_pupil_pos(times, stimStarts, pupilXy)
    f, ax = plt.subplots(1)
    aa = pupilXy_trials[prevFeedback == 1, :]
    ax.scatter(aa[:, 0], aa[:, 1], facecolor="b")
    aa = pupilXy_trials[prevFeedback == 0, :]
    ax.scatter(aa[:, 0], aa[:, 1], facecolor="r")
    ax.set_xlabel("Pupil X position")
    ax.set_ylabel("Pupil Y position")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.legend(["Prev. Reward", " Prev. No Reward"])
    nanInd = np.where(
        np.isnan(pupilXy_trials[:, 0]) & np.isnan(pupilXy_trials[:, 1])
    )[0]

    if (
        np.any((pupilXy[:, 0] > 210))
        & np.any((pupilXy[:, 1] > 135))
        & np.any((pupilXy[:, 1] > 135) & (pupilXy[:, 0] > 180))
    ):
        print(str(n))

#%% Panel E
plt.close("all")
pvals = []
N = len(AllData)
meansP = []
meansN = []
meanPs = []
meanNs = []
posnegalt = []
posnegaz = []
ts = []
for n in range(1, len(AllData) + 1):
    d = AllData[n]
    stimStarts = d["stimT"][:, 0]
    fbType = d["feedbackType"]
    prevReward = np.append(0, fbType[:-1, 0])
    pupilXy = d["pupilxy"]
    # pupilXy/=np.nanmax(pupilXy,0)
    times = d["pupilTimes"]
    window = np.array([-0.01, 1]).reshape(-1, 1).T
    pup, t = AlignStim(
        pupilXy, times, stimStarts, window, timeUnit=1, timeLimit=1
    )
    ts.append(t)
    # find WheelMovement indices to remove
    wh, tt = AlignStim(
        AllData[n]["wheelVelocity"],
        AllData[n]["wheelTimes"],
        stimStarts,
        window,
    )

    wh = wh[((tt <= 0.5) & (tt >= 0)), :, 0]
    wh = np.nanmean(np.abs(wh), 0)
    notMoveDuringStimInd = np.where(wh == 0)[0]

    pup_P = pup[:, (prevReward == 1) & (wh == 0), :]
    pup_N = pup[:, (prevReward == 0) & (wh == 0), :]

    # Pupil
    pupil_ds = AlignSignals(
        d["pupil"],
        d["pupilTimes"],
        d["calTimes"],
    )

    # make into distance
    pup_P -= pup_P[t == 0, :, :]
    pup_N -= pup_N[t == 0, :, :]
    distP = np.sqrt(np.sum(pup_P**2, 2))
    distN = np.sqrt(np.sum(pup_N**2, 2))

    mp = np.nanmean(distP, 1)
    sp = np.nanstd(distP, 1) / np.sqrt(distP.shape[1])
    mn = np.nanmean(distN, 1)
    sn = np.nanstd(distN, 1) / np.sqrt(distN.shape[1])

    meansP.append(np.nanmean(mp[(t >= 0) & (t <= 0.5)]))
    meansN.append(np.nanmean(mn[(t >= 0) & (t <= 0.5)]))
    # METHOD 1
    # pup_P[:, 0] = np.abs(pup_P[:, 0])
    # pup_N[:, 0] = np.abs(pup_N[:, 0])
    # pup_P[:, 1] = np.abs(pup_P[:, 1])
    # pup_N[:, 1] = np.abs(pup_N[:, 1])
    # pup_P[:, 0] = np.abs(pup_P[:, 0])
    # pup_N[:, 0] = np.abs(pup_N[:, 0])
    # pup_P[:, 1] = np.abs(pup_P[:, 1])
    # pup_N[:, 1] = np.abs(pup_N[:, 1])
    mp = np.nanmedian(pup_P, 1)
    meanPs.append(mp)
    sp = np.nanstd(pup_P, 1) / np.sqrt(pup_P.shape[1])
    mn = np.nanmedian(pup_N, 1)
    meanNs.append(mn)
    sn = np.nanstd(pup_N, 1) / np.sqrt(pup_N.shape[1])

    posnegaz.append(
        (
            np.nanmean(np.nanmean(pup_P[(t >= 0) & (t <= 0.5), :, 0], 0)),
            np.nanmean(np.nanmean(pup_N[(t >= 0) & (t <= 0.5), :, 0], 0)),
        )
    )
    posnegalt.append(
        (
            np.nanmean(np.nanmean(pup_P[(t >= 0) & (t <= 0.5), :, 0], 1)),
            np.nanmean(np.nanmean(pup_N[(t >= 0) & (t <= 0.5), :, 1], 0)),
        )
    )

slowestT = []
minSamples = 1000
for tt in ts:
    dur = len(tt)
    if dur < minSamples:
        minSamples = dur
        slowestT = tt  # this will be what others will downsample to


t = slowestT
for i in range(len(meanPs)):
    if len(ts[i]) == minSamples:
        continue
    meanPs[i] = AlignSignals(meanPs[i], ts[i], slowestT, False)
    meanNs[i] = AlignSignals(meanNs[i], ts[i], slowestT, False)

meanPs = np.dstack(meanPs)
meanNs = np.dstack(meanNs)

f, ax = plt.subplots(2, 1, sharex=True, sharey=True)
mp = np.nanmedian(meanPs, -1)
sp = np.nanstd(meanPs, -1) / np.sqrt(21)
mn = np.nanmedian(meanNs, -1)
sn = np.nanstd(meanNs, -1) / np.sqrt(21)
ax[0].plot(t, mp[:, 0], "b")
ax[0].fill_between(
    t, mp[:, 0] - sp[:, 0], mp[:, 0] + sp[:, 0], color="b", alpha=0.4
)
ax[0].plot(t, mn[:, 0], "r")
ax[0].fill_between(
    t, mn[:, 0] - sn[:, 0], mn[:, 0] + sn[:, 0], color="r", alpha=0.4
)


ax[1].plot(t, mp[:, 1], "b")
ax[1].fill_between(
    t, mp[:, 1] - sp[:, 1], mp[:, 1] + sp[:, 1], color="b", alpha=0.4
)
ax[1].plot(t, mn[:, 1], "r")
ax[1].fill_between(
    t, mn[:, 1] - sn[:, 1], mn[:, 1] + sn[:, 1], color="r", alpha=0.4
)
ax[0].set_xlim(0, 0.5)
ax[0].set_ylim(-0.5, 0.5)
ax[1].set_xlabel("time")
ax[0].set_ylabel("distance")
ax[1].set_ylabel("distance")
ax[0].spines["right"].set_visible(False)
ax[0].spines["top"].set_visible(False)
ax[0].set_aspect(1.0 / ax[0].get_data_ratio(), adjustable="box")
ax[1].spines["right"].set_visible(False)
ax[1].spines["top"].set_visible(False)
ax[1].set_aspect(1.0 / ax[1].get_data_ratio(), adjustable="box")

#%% Panel F-G
plt.close("all")
lck = ShowLickingForState(AllData, influence="feedback")
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
    percPos.append(np.sum((l[indPostPos] > 0) / len(l[indPrePos])) * 100)
    percNeg.append(np.sum((l[indPostNeg] > 0) / len(l[1])) * 100)

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

df = pd.DataFrame(
    {
        "Pupil": np.hstack(
            (np.repeat(["Large"], 22 * 2), (np.repeat(["Small"], 22 * 2)))
        ),
        "Period": np.tile(
            np.hstack((np.repeat(["pre"], 22), np.repeat(["post"], 22))),
            (1, 2),
        )[0, :],
        "licks": np.hstack((NumPospre, NumPos, NumNegpre, NumNeg)),
    }
)
a = sns.catplot(
    x="Period",
    y="licks",
    hue="Pupil",
    data=df,
    kind="point",
    palette=["b", "r"],
)
plt.xlabel("Licks (Hz)")
plt.tight_layout()

#%% Panel J + K
movieDir = "Z:\\UCLData\\Ephys_Task\\Subjects"
dataDir = "D:\\ProjectTask\\Ephys"

processedFiles = glob.glob(
    os.path.join(movieDir, "**/*_proc.npy"), recursive=True
)

fbTypes = []
fbTimes = []
stimTimes = []
lockedS = []
lockedF = []
motSvds = []
times = []
ts = []
pupClass = []
for i in range(len(processedFiles)):
    try:
        movieFile = processedFiles[i]
        # open svd data (motSvd) and the first dimension thereof
        proc = np.load(movieFile, allow_pickle=True).item()
        # motSvd = proc["motSVD"][-1][:, 0].reshape(-1, 1)
        motSvd = (proc["motSVD"][-1] @ proc["motSv"]).reshape(-1, 1)
        # motSvd = sp.stats.zscore(motSvd, nan_policy="omit")
        motSvd = sp.signal.medfilt(motSvd, (11, 1))
        motSvd -= np.nanmin(motSvd)
        motSvd /= np.nanmax(motSvd)
        # get directory path
        directoryStructure = os.path.split(movieFile)[0]
        dirPath = os.path.split(directoryStructure.removeprefix(movieDir))[0]
        dataDirSpecific = dataDir + dirPath

        # load important variables
        # faceTime = sp.io.loadmat(
        #     os.path.join(directoryStructure, "face_timeStamps.mat")
        # )["tVid"]
        faceTime = np.load(
            os.path.join(directoryStructure, "face.timestamps.npy")
        )
        stimT = np.load(
            os.path.join(dataDirSpecific, "_ss_trials.stimOn_intervals.npy")
        )[:, 0].reshape(-1, 1)
        fbType = np.load(
            os.path.join(dataDirSpecific, "_ss_trials.feedbackType.npy")
        )
        fbTime = np.load(
            os.path.join(dataDirSpecific, "_ss_trials.feedback_times.npy")
        )

        pupildia = np.load(os.path.join(dataDirSpecific, "eye.diameter.npy"))
        pupilTimes = np.load(
            os.path.join(dataDirSpecific, "eye.timestamps.npy")
        )

        wheelVelocity = np.load(
            os.path.join(
                dataDirSpecific,
                "_ss_wheel.velocity.npy",
            )
        )
        recTimes = np.load(
            os.path.join(
                dataDirSpecific,
                "_ss_signals.timestamps.npy",
            )
        )
        window = np.array([-0.5, 2], ndmin=2)

        # Pupil

        tind = np.where(
            (pupilTimes >= stimT[0]) & (pupilTimes <= (stimT[-1] + 5000))
        )[0]
        middle = np.nanmedian(pupildia[tind, 0])
        pu, t = AlignStim(
            pupildia, pupilTimes, stimT, np.array([-0.2, 1], ndmin=2)
        )
        bigTrials = pu > middle
        bigTrialsSum = np.sum(bigTrials[:, :, 0], 0)
        meanPupil = np.nanmean(pu[:, :, 0], 0)
        largePupil = (bigTrialsSum / len(t)) > 0.5
        smallPupil = (bigTrialsSum / len(t)) < 0.5

        ##### remove wheel movement times
        # find WheelMovement indices to remove
        wh, t = AlignStim(
            wheelVelocity,
            recTimes,
            stimT,
            np.array([-0.5, 0.5], ndmin=2),
        )
        wh = wh[((t <= 0.5) & (t >= 0)), :, 0]
        wh = np.nanmean(np.abs(wh), 0)
        moveDuringStimInd = np.where(wh > 0)[0]
        #################################

        whS = AlignStim(motSvd, faceTime, stimT, window)
        whF = AlignStim(motSvd, faceTime, fbTime, window)
        wht = whS[1]
        whS = whS[0][:, :, 0]
        whF = whF[0][:, :, 0]

        fbTypes.append(np.delete(fbType, moveDuringStimInd, axis=0))
        pupClass.append(np.delete(largePupil, moveDuringStimInd, axis=0))
        lockedS.append(np.delete(whS, moveDuringStimInd, axis=1))
        lockedF.append(np.delete(whF, moveDuringStimInd, axis=1))
        motSvds.append(motSvd)
        ts.append(wht)
        fbTimes.append(np.delete(fbTime, moveDuringStimInd, axis=0))
        stimTimes.append(np.delete(stimT, moveDuringStimInd, axis=0))
        times.append(faceTime)
    except:
        print("error in " + movieFile)
        print(print(traceback.format_exc()))
        continue

plt.close("all")
N = len(lockedF)

# shape: s/f X N X prevFB
befores = np.zeros((2, N, 2))
afters = np.zeros((2, N, 2))

diffs = np.zeros((2, N))

stimPos = []
stimNeg = []
fbPos = []
fbNeg = []
for i in range(N):
    whS = lockedS[i]
    whF = lockedF[i]
    fb = fbTypes[i][:, 0]
    prevfb = np.append(0, fb[:-1])
    #####change if want to look at pupuil
    prevfb = pupClass[i]
    #######################
    t = ts[i]
    bs = np.nanmean(whS[(t < 0), :], 0)
    bf = np.nanmean(whF[(t < 0), :], 0)
    afs = np.nanmean(whS[(t >= 0) & (t <= 0.5), :], 0)
    aff = np.nanmean(whF[(t >= 0) & (t <= 0.5), :], 0)

    befores[0, i, 0] = np.nanmean(bs[prevfb == 0])
    befores[0, i, 1] = np.nanmean(bs[prevfb == 1])
    befores[1, i, 0] = np.nanmean(bf[prevfb == 0])
    befores[1, i, 1] = np.nanmean(bf[prevfb == 1])

    afters[0, i, 0] = np.nanmean(afs[prevfb == 0])
    afters[0, i, 1] = np.nanmean(afs[prevfb == 1])
    afters[1, i, 0] = np.nanmean(aff[prevfb == 0])
    afters[1, i, 1] = np.nanmean(aff[prevfb == 1])

    generalAvg = np.nanmean(
        whS[(t >= 0) & (t <= 0.5), :] - np.nanmean(whS[(t < 0), :], 0), 0
    )
    diffs[0, i] = np.nanmean(generalAvg[prevfb == 0])
    diffs[1, i] = np.nanmean(generalAvg[prevfb == 1])

    m = np.nanmean(whS[:, prevfb == 0], 1)
    stimNeg.append(m)
    se = np.nanstd(whS[:, prevfb == 0], 1) / np.sqrt(N)

    m = np.nanmean(whS[:, prevfb == 1], 1)
    stimPos.append(m)

    se = np.nanstd(whS[:, prevfb == 1], 1) / np.sqrt(N)

    m = np.nanmean(whF[:, fb == 0], 1)
    fbNeg.append(m)

    se = np.nanstd(whF[:, fb == 0], 1) / np.sqrt(N)

    m = np.nanmean(whF[:, fb == 1], 1)
    fbPos.append(m)

    se = np.nanstd(whF[:, fb == 1], 1) / np.sqrt(N)


pfb = np.hstack(
    (
        np.repeat(["negative"], befores.shape[1]),
        np.repeat(["positive"], befores.shape[1]),
    )
)
pfb = np.hstack((pfb, pfb))
time = np.append(
    np.repeat(["before"], befores.shape[1] * 2),
    np.repeat(["after"], befores.shape[1] * 2),
)
ids = np.repeat(np.arange(0, befores.shape[1]), 4)
vals = np.hstack(
    (befores[0, :, 0], befores[0, :, 1], afters[0, :, 0], afters[0, :, 1])
)

df = pd.DataFrame({"ids": ids, "pfb": pfb, "time": time, "vals": vals})

negResp = (
    df[(df.pfb == "negative") & (df.time == "after")].vals.to_numpy()
    - df[(df.pfb == "negative") & (df.time == "before")].vals.to_numpy()
)
posResp = (
    df[(df.pfb == "positive") & (df.time == "after")].vals.to_numpy()
    - df[(df.pfb == "positive") & (df.time == "before")].vals.to_numpy()
)

stimPos = np.vstack(stimPos)
stimNeg = np.vstack(stimNeg)
fbPos = np.vstack(fbPos)
fbNeg = np.vstack(fbNeg)


N = stimPos.shape[0]
f, ax = plt.subplots(2, sharex=True, sharey=True)
m = np.nanmean(stimNeg, 0)
se = np.nanmean(stimNeg, 0) / np.sqrt(N)
ax[0].plot(t, m, "r")
ax[0].fill_between(t, m - se, m + se, color="r", alpha=0.5)
m = np.nanmean(stimPos, 0)
ax[0].plot(t, m, "b")
se = np.nanmean(stimPos, 0) / np.sqrt(N)
ax[0].fill_between(t, m - se, m + se, color="b", alpha=0.5)
m = np.nanmean(fbNeg, 0)
ax[1].plot(t, m, "r")
se = np.nanmean(fbNeg, 0) / np.sqrt(N)
ax[1].fill_between(t, m - se, m + se, color="r", alpha=0.5)
m = np.nanmean(fbPos, 0)
ax[1].plot(t, m, "b")
se = np.nanmean(fbPos, 0) / np.sqrt(N)
ax[1].fill_between(t, m - se, m + se, color="b", alpha=0.5)
ax[0].set_title("WHisking Energy locked to stim")
ax[0].set_xlabel("time from stim (s)")
ax[1].set_title("WHisking Energy locked to feedback")
ax[1].set_xlabel("time from feedback (s)")
ax[0].set_xlim(-0.5, 0.5)
ax[0].spines["right"].set_visible(False)
ax[0].spines["top"].set_visible(False)
ax[0].set_aspect(1.0 / ax[0].get_data_ratio(), adjustable="box")

ax[1].spines["right"].set_visible(False)
ax[1].spines["top"].set_visible(False)
ax[1].set_aspect(1.0 / ax[1].get_data_ratio(), adjustable="box")


prePos = np.nanmean(stimPos[:, t <= 0], 1)
preNeg = np.nanmean(stimNeg[:, t <= 0], 1)
postPos = np.nanmean(stimPos[:, (t >= 0) & (t <= 0.5)], 1)
postNeg = np.nanmean(stimNeg[:, (t >= 0) & (t <= 0.5)], 1)

df = pd.DataFrame(
    {
        "fb": np.hstack(
            (np.repeat(["positive"], N * 2), (np.repeat(["negative"], N * 2)))
        ),
        "Period": np.tile(
            np.hstack((np.repeat(["pre"], N), np.repeat(["post"], N))),
            (1, 2),
        )[0, :],
        "whisking": np.hstack((prePos, postPos, preNeg, postNeg)),
    }
)

a = sns.catplot(
    x="Period",
    y="whisking",
    hue="fb",
    data=df,
    kind="point",
    palette=["b", "r"],
    errorbar="se",
)
plt.xlabel("Licks (Hz)")
plt.tight_layout()

#%% Panel H+I
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
