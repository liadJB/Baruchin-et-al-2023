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
nullProps1 = np.load(
    "D:\\Figures\\Paper_Feedback\\Fitting\\Cumulative\\nulProps1.npy"
)
nullProps2 = np.load(
    "D:\\Figures\\Paper_Feedback\\Fitting\\Cumulative\\nulProps2.npy"
)
nullProps3 = np.load(
    "D:\\Figures\\Paper_Feedback\\Fitting\\Cumulative\\nulProps3.npy"
)
nullProps4 = np.load(
    "D:\\Figures\\Paper_Feedback\\Fitting\\Cumulative\\nulProps4.npy"
)

#%% Panel A
PrintResponseByAllGroups(AllData, iteration=[16], nn=[136], saveDir=None)

#%% Panel B
fits = np.array(modelsBase["fit"][modelsBase["Id"] == (16, 136)])
GetFitFromFitList(fits, "F")

#%% Panel C
PrintResponseByAllGroups(AllData, iteration=[16], nn=[65], saveDir=None)

#%% Panel D
fits = np.array(modelsBase["fit"][modelsBase["Id"] == (16, 65)])
GetFitFromFitList(fits, "P")

#%% Panel E


colours = np.zeros((256, 4))
red = np.ones(256)
green = np.append(np.ones(86), np.linspace(1, 0.3, 256 - 86))

blue = np.append(np.linspace(1, 0.1, 86), np.ones(256 - 86) * 0)

alphas = np.ones_like(red)
colours[:, 0] = red
colours[: len(green), 1] = green
colours[: len(blue), 2] = blue
colours[:, -1] = alphas
newcmp = ListedColormap(colours)


plt.close("all")

tups = np.vstack((sessId, nId)).T
tups = tups[r[0, :] == 1, :]
modelsBase = GetOneFunctionVarianceDifferentModels(
    dataDir="D:\\Figures\\OneFunction_raw\\",
    cutoff=0.01,
    shuffles=True,
    resp=tups,
)
f1 = np.append(
    np.array(modelsBase[modelsBase["bestFit"] == "F"]["Id"]),
    np.array(modelsBase[modelsBase["bestFit"] == "PF"]["Id"]),
)
f1 = np.vstack(f1)
vals, counts = np.unique(f1[:, 0], return_counts=True)
vals -= 1
# count neurons


prevPosResp = []
prevNegResp = []
t_list = []
sessIds = []
cr = 0
for i in range(1, len(AllData) + 1):
    if not (i in f1[:, 0]):
        continue
    relNeurons = f1[f1[:, 0] == i, 1]

    d = AllData[i]
    sig = d["calTrace"]
    # normalise sig
    # sig/=np.nanmax(sig,axis = 0)
    delays = d["planeDelays"]
    planes = d["planes"]
    times = d["calTimes"]
    imagingSide = d["ImagingSide"]

    stimStarts = d["stimT"][:, 0]
    stimEnds = d["stimT"][:, 1]

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
    prevFeedback = np.append(0, feedbackType[1:])
    window = np.array([[-0.5, 1]])
    times_new = np.arange(times[0], times[-1], 0.01)
    sig = sig[:, relNeurons]
    planes = planes[relNeurons, :]
    sig_new = np.zeros((len(times_new), sig.shape[-1]))

    for iii in range(sig_new.shape[-1]):
        ff = sp.interpolate.interp1d(times[:, 0], sig[:, iii])
        sig_new[:, iii] = ff(times_new)
    sig = sig_new
    times = times_new
    stimA, t = GetCalciumAligned(
        sig, times, stimStarts, window, planes, delays
    )

    cr = cr + stimA.shape[-1]
    pre = np.nanmean(stimA[t <= 0, :, :], 0)
    post = np.nanmean(stimA[(t > 0.2) & (t <= 1), :, :], 0)

    flipInds = np.where(np.nanmean(post, 0) < np.nanmean(pre, 0))[0]
    if len(flipInds) > 0:
        stimA[:, :, flipInds] = -stimA[:, :, flipInds]

    stimA -= np.nanmean(
        stimA[(t <= 0), :, :],
        0,
    )

    prevPosResp.append(
        np.nanmean(
            stimA[:, (contra == np.nanmax(contra)) & (prevFeedback == 1), :], 1
        )
    )
    prevNegResp.append(
        np.nanmean(
            stimA[:, (contra == np.nanmax(contra)) & (prevFeedback == 0), :], 1
        )
    )
    t_list.append(t)
    sessIds.append(np.ones(stimA.shape[-1]) * i)

# Need to downsample some signals
minSamples = 1000
slowestT = []
for tt in t_list:
    dur = len(tt)
    if dur < minSamples:
        minSamples = dur
        slowestT = tt  # this will be what others will downsample to
        t = slowestT

for i in range(len(prevPosResp)):
    if len(t_list[i]) == minSamples:
        continue
    prevPosResp[i] = AlignSignals(prevPosResp[i], t_list[i], slowestT, False)
    prevNegResp[i] = AlignSignals(prevNegResp[i], t_list[i], slowestT, False)

prevPosResp = np.hstack(prevPosResp)
prevNegResp = np.hstack(prevNegResp)
sessIds = np.hstack(sessIds)

ratio = prevPosResp / prevNegResp

meanRatio = np.nanmean(ratio[(slowestT >= 0) & (slowestT <= 0.5)], axis=0)

df = (t[1] - t[0]) / 2

prevNegResp = sp.signal.medfilt(prevNegResp, (3, 1))
prevPosResp = sp.signal.medfilt(prevPosResp, (3, 1))

pre = np.nanmean(prevNegResp[t <= 0, :], 0)
post = np.nanmean(prevNegResp[(t > 0) & (t <= 0.5), :], 0)
flipInds = np.where(post < pre)[0]


### second option
t_new = np.arange(-0.2, 1, 0.01)


negMax = np.nanmean(prevNegResp[(slowestT >= 0) & (slowestT <= 1)], 0)
negNorm = prevNegResp / negMax
posNorm = prevPosResp / negMax


meanRatio = np.nanmedian(posNorm[(slowestT >= 0) & (slowestT <= 0.5)], axis=0)


f, ax = plt.subplots(1, 2, sharex=True)
im = ax[0].imshow(
    np.flip(negNorm[:, np.argsort(meanRatio)].T, axis=0),
    cmap=newcmp,  # "YlOrRd",
    vmin=0,
    vmax=3,
    extent=(slowestT[0] + df, slowestT[-1] + df, 0, ratio.shape[1]),
    aspect="auto",
)
im = ax[1].imshow(
    np.flip(posNorm[:, np.argsort(meanRatio)].T, axis=0),
    cmap=newcmp,  # "YlOrRd",
    vmin=0,
    vmax=3,
    extent=(slowestT[0] + df, slowestT[-1] + df, 0, ratio.shape[1]),
    aspect="auto",
)
ax[1].set_xlim(-0.25, 1)
f.colorbar(im)

#%% Panel F

colours = np.zeros((256, 4))
red = np.ones(256)
green = np.append(np.ones(86), np.linspace(1, 0.3, 256 - 86))  # np.append(

blue = np.append(np.linspace(1, 0.1, 86), np.ones(256 - 86) * 0)  # np.append(

alphas = np.ones_like(red)
colours[:, 0] = red
colours[: len(green), 1] = green
colours[: len(blue), 2] = blue
colours[:, -1] = alphas
newcmp = ListedColormap(colours)


tups = np.vstack((sessId, nId)).T
tups = tups[r[0, :] == 1, :]
modelsBase = GetOneFunctionVarianceDifferentModels(
    dataDir="D:\\Figures\\OneFunction_raw\\",
    cutoff=0.01,
    shuffles=True,
    resp=tups,
)
f1 = np.append(
    np.array(modelsBase[modelsBase["bestFit"] == "P"]["Id"]),
    np.array(modelsBase[modelsBase["bestFit"] == "PF"]["Id"]),
)
f1 = np.vstack(f1)
vals, counts = np.unique(f1[:, 0], return_counts=True)
vals -= 1
# count neurons
window = np.array([[-0.2, 1]])

prevPosResp = []
prevNegResp = []
t_list = []
for i in range(1, len(AllData) + 1):
    if not (i in f1[:, 0]):
        continue
    relNeurons = f1[f1[:, 0] == i, 1]

    d = AllData[i]
    sig = d["calTrace"]
    # normalise sig
    # sig/=np.nanmax(sig,axis = 0)
    delays = d["planeDelays"]
    planes = d["planes"]
    times = d["calTimes"]
    imagingSide = d["ImagingSide"]

    stimStarts = d["stimT"][:, 0]
    stimEnds = d["stimT"][:, 1]

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
    prevFeedback = np.append(0, feedbackType[1:])

    # Pupil
    pupil_ds = AlignSignals(
        d["pupil"],
        d["pupilTimes"],
        d["calTimes"],
    )
    middle = np.nanmedian(pupil_ds, 0)
    pu, t = AlignStim(pupil_ds, times, stimStarts, window)
    bigTrials = pu > middle
    bigTrialsSum = np.sum(bigTrials[:, :, 0], 0)
    meanPupil = np.nanmean(pu[:, :, 0], 0)
    largePupil = (bigTrialsSum / len(t)) > 0.5
    smallPupil = (bigTrialsSum / len(t)) < 0.5

    times_new = np.arange(times[0], times[-1], 0.01)
    sig = sig[:, relNeurons]
    planes = planes[relNeurons, :]
    sig_new = np.zeros((len(times_new), sig.shape[-1]))

    for iii in range(sig_new.shape[-1]):
        ff = sp.interpolate.interp1d(times[:, 0], sig[:, iii])
        sig_new[:, iii] = ff(
            times_new
        )  # np.interp(times_new,sig[:,iii],times[:,0])
    sig = sig_new
    times = times_new
    stimA, t = GetCalciumAligned(
        sig, times, stimStarts, window, planes, delays
    )
    # stimA = stimA[:, :, relNeurons]
    window = np.array([[-0.5, 1]])
    stimA, t = GetCalciumAligned(
        sig, times, stimStarts, window, planes, delays
    )
    # stimA = stimA[:, :, relNeurons]
    pre = np.nanmean(stimA[t <= 0, :, :], 0)
    post = np.nanmean(stimA[(t > 0.2) & (t <= 1), :, :], 0)
    flipInds = np.where(np.nanmean(post, 0) < np.nanmean(pre, 0))[0]
    if len(flipInds) > 0:
        stimA[:, :, flipInds] = -stimA[:, :, flipInds]

    stimA -= np.nanmean(
        stimA[(t <= 0), :, :],
        0,
    )

    prevPosResp.append(
        np.nanmean(
            stimA[:, (contra == np.nanmax(contra)) & (largePupil == 1), :], 1
        )
    )
    prevNegResp.append(
        np.nanmean(
            stimA[:, (contra == np.nanmax(contra)) & (largePupil == 0), :], 1
        )
    )
    t_list.append(t)

# Need to downsample some signals
minSamples = 1000
slowestT = []
for tt in t_list:
    dur = len(tt)
    if dur < minSamples:
        minSamples = dur
        slowestT = tt  # this will be what others will downsample to
        t = slowestT

for i in range(len(prevPosResp)):
    if len(t_list[i]) == minSamples:
        continue
    prevPosResp[i] = AlignSignals(prevPosResp[i], t_list[i], slowestT, False)
    prevNegResp[i] = AlignSignals(prevNegResp[i], t_list[i], slowestT, False)

prevPosResp = np.hstack(prevPosResp)
prevNegResp = np.hstack(prevNegResp)

ratio = prevPosResp / prevNegResp

meanRatio = np.nanmean(ratio[(slowestT >= 0) & (slowestT <= 1)], axis=0)

df = (t[1] - t[0]) / 2


prevNegResp = sp.signal.medfilt(prevNegResp, (3, 1))
prevPosResp = sp.signal.medfilt(prevPosResp, (3, 1))


### second option
negMax = np.nanmean(prevNegResp[(slowestT >= 0) & (slowestT <= 1)], 0)
negNorm = prevNegResp / negMax
posNorm = prevPosResp / negMax

meanRatio = np.nanmedian(posNorm[(slowestT >= 0) & (slowestT <= 0.5)], axis=0)


f, ax = plt.subplots(1, 2, sharex=True)
im = ax[0].imshow(
    np.flip(negNorm[:, np.argsort(meanRatio)].T, axis=0),
    cmap=newcmp,
    vmin=0,
    vmax=3,
    extent=(slowestT[0] + df, slowestT[-1] + df, 0, ratio.shape[1]),
    aspect="auto",
)
im = ax[1].imshow(
    np.flip(posNorm[:, np.argsort(meanRatio)].T, axis=0),
    cmap=newcmp,
    vmin=0,
    vmax=3,
    extent=(slowestT[0] + df, slowestT[-1] + df, 0, ratio.shape[1]),
    aspect="auto",
)

ax[1].set_xlim(-0.25, 1)
f.colorbar(im)

#%% Panels G-N
plt.close("all")


def CreateNullIntervalPlot(ax, dist):
    ys = []
    xs = []
    f, axt = plt.subplots(1)
    # sns.kdeplot(data=dist.T,legend=False,common_norm=False,cumulative=True,ax=axt,clip=(-2,2))
    sns.kdeplot(
        data=dist,
        ax=axt,
        legend=False,
        common_norm=False,
        common_grid=True,
        cumulative=True,
        clip=(-2, 2),
    )

    for i in range(0, len(axt.lines)):
        line = axt.lines[i]
        x, y = line.get_data()
        y /= np.max(y)
        ys.append(y)
        xs.append(x)
        plt.close(f)
    ys = np.vstack(ys)
    xs = np.vstack(xs)

    ax.fill_between(
        x,
        np.nanpercentile(ys, 2.5, axis=0),
        np.nanpercentile(ys, 97.5, axis=0),
        color="grey",
        alpha=0.4,
    )

    return None


# Find the R for pupil and feedback
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

##### Feedback Plotting #########
f, ax = plt.subplots(1)
ax.scatter(nfr[:, 0] + nfr[:, 1], nfr[:, 0], s=30, c=["w"], edgecolors="grey")
ax.scatter(fr[:, 0] + fr[:, 1], fr[:, 0], s=30, c=["k"], edgecolors="k")
ax.scatter(fpr[:, 0] + fpr[:, 2], fpr[:, 0], s=30, c=["k"], edgecolors="k")
ax.set_aspect("equal", adjustable="box")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.plot(np.arange(0, 0.7, 0.1), np.arange(0, 0.7, 0.1), "k--")
ax.set_title("R (gain) feedback")
ax.set_xlabel("Positive Feedback")
ax.set_ylabel("Negative Feedback")
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
f.set_size_inches(8, 8)

# Extra histogram to put diagonally
f, ax = plt.subplots(1)

valsPos_ns = nullProps1[:, :, [0, 4]]
valsPos_ns = valsPos_ns[:, :, 1] / (
    0.5 * (2 * valsPos_ns[:, :, 0] + valsPos_ns[:, :, 1])
)
valsPos = (frall[:, 0] + frall[:, 1] - frall[:, 0]) / (
    0.5 * (frall[:, 0] + frall[:, 1] + frall[:, 0])
)

sns.kdeplot(
    data=[valsPos],
    palette=["black"],
    ax=ax,
    legend=False,
    common_norm=False,
    cumulative=True,
    clip=(-2, 2),
)
sorted_data = np.sort(valsPos)
y = np.arange(sorted_data.size)
y = y / np.max(y)
CreateNullIntervalPlot(ax, valsPos_ns)
plt.grid(False)
ax.set_xlabel("Respone Modulation (%)")
ax.set_ylabel("Proportion of Neurons")
ax.set_xlim(-2, 2)
ax.set_ylim(0, 1)
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
f.set_size_inches(np.sqrt(8), np.sqrt(8))
ax.vlines(0, 0, 1, "k")
##### Pupil Plotting #########
f, ax = plt.subplots(1)
ax.scatter(npr[:, 0] + npr[:, 1], npr[:, 0], s=30, c=["w"], edgecolors="grey")
ax.scatter(pr[:, 0] + pr[:, 1], pr[:, 0], s=30, c=["k"], edgecolors="k")
ax.scatter(fpr[:, 0] + fpr[:, 1], fpr[:, 0], s=30, c=["k"], edgecolors="k")
ax.set_aspect("equal", adjustable="box")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.plot(np.arange(0, 0.7, 0.1), np.arange(0, 0.7, 0.1), "k--")
ax.set_title("R (gain) Pupil")
ax.set_xlabel("Large Pupil")
ax.set_ylabel("Small Pupil")
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
f.set_size_inches(8, 8)

# Extra histogram to put diagonally
f, ax = plt.subplots(1)

valsPos_ns = nullProps2[:, :, [0, 1]]
valsPos_ns = valsPos_ns[:, :, 1] / (
    0.5 * (2 * valsPos_ns[:, :, 0] + valsPos_ns[:, :, 1])
)
valsPos = (prall[:, 1]) / (0.5 * (2 * prall[:, 0] + prall[:, 1]))
sns.kdeplot(
    data=[valsPos],
    palette=["black"],
    ax=ax,
    legend=False,
    common_norm=False,
    cumulative=True,
    clip=(-2, 2),
)

CreateNullIntervalPlot(ax, valsPos_ns)
plt.grid(False)
ax.set_xlabel("Respone Modulation (%)")
ax.set_ylabel("Proportion of Neurons")
ax.set_xlim(-2, 2)
ax.set_ylim(0, 1)
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
f.set_size_inches(np.sqrt(8), np.sqrt(8))
ax.vlines(0, 0, 1, "k")


f, ax = plt.subplots(1)
ax.scatter(np.abs(fpr[:, 1]), np.abs(fpr[:, 2]), s=30, c=["k"], edgecolors="k")
ax.set_aspect("equal", adjustable="box")
ax.set_xlim(0, 0.45)
ax.set_ylim(0, 0.45)
ax.plot(np.arange(-0.7, 0.7, 0.1), np.arange(-0.7, 0.7, 0.1), "k--")
ax.set_title("Pupil gain vs. Feeback gain")
ax.set_xlabel("Pupil gain")
ax.set_ylabel("Feedback gain")
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)

##### GoNoGo Plotting #########
f, ax = plt.subplots(1)
ax.scatter(ngr[:, 0] + ngr[:, 1], ngr[:, 0], s=30, c=["w"], edgecolors="grey")
ax.scatter(gr[:, 0] + gr[:, 1], gr[:, 0], s=30, c=["k"], edgecolors="k")
ax.set_aspect("equal", adjustable="box")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.plot(np.arange(0, 0.7, 0.1), np.arange(0, 0.7, 0.1), "k--")
ax.set_title("R (gain) Go")
ax.set_xlabel("Go")
ax.set_ylabel("No Go")
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
f.set_size_inches(8, 8)

# Extra histogram to put diagonally
f, ax = plt.subplots(1)

valsPos_ns = nullProps3[:, :, [0, 2]]
valsPos_ns = valsPos_ns[:, :, 1] / (
    0.5 * (2 * valsPos_ns[:, :, 0] + valsPos_ns[:, :, 1])
)
valsPos = (grall[:, 1]) / (0.5 * (2 * grall[:, 0] + grall[:, 1]))
sns.kdeplot(
    data=[valsPos],
    palette=["black"],
    ax=ax,
    legend=False,
    common_norm=False,
    cumulative=True,
    clip=(-2, 2),
)

CreateNullIntervalPlot(ax, valsPos_ns)
plt.grid(False)
ax.set_xlabel("Respone Modulation (%)")
ax.set_ylabel("Proportion of Neurons")
ax.set_xlim(-2, 2)
ax.set_ylim(0, 1)
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
f.set_size_inches(np.sqrt(8), np.sqrt(8))
ax.vlines(0, 0, 1, "k")
##### Correctness Plotting #########
f, ax = plt.subplots(1)
ax.scatter(ncr[:, 0] + ncr[:, 1], ncr[:, 0], s=30, c=["w"], edgecolors="grey")
ax.scatter(cr[:, 0] + cr[:, 1], cr[:, 0], s=30, c=["k"], edgecolors="k")
ax.set_aspect("equal", adjustable="box")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.plot(np.arange(0, 0.7, 0.1), np.arange(0, 0.7, 0.1), "k--")
ax.set_title("R (gain) Correct")
ax.set_xlabel("Correct")
ax.set_ylabel("Incorrect")
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
f.set_size_inches(8, 8)

# Extra histogram to put diagonally
f, ax = plt.subplots(1)

valsPos_ns = nullProps4[:, :, [0, 3]]
valsPos_ns = valsPos_ns[:, :, 1] / (
    0.5 * (2 * valsPos_ns[:, :, 0] + valsPos_ns[:, :, 1])
)
valsPos = (crall[:, 1]) / (0.5 * (2 * crall[:, 0] + crall[:, 1]))
sns.kdeplot(
    data=[valsPos],
    palette=["black"],
    ax=ax,
    legend=False,
    common_norm=False,
    cumulative=True,
    clip=(-2, 2),
)

CreateNullIntervalPlot(ax, valsPos_ns)
plt.grid(False)
ax.set_xlabel("Respone Modulation (%)")
ax.set_ylabel("Proportion of Neurons")
ax.set_xlim(-2, 2)
ax.set_ylim(0, 1)
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
f.set_size_inches(np.sqrt(8), np.sqrt(8))
ax.vlines(0, 0, 1, "k")
