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

AllData, AllSubjects, AllSessions = LoadAllFiles("D:\\task_2p\\")
for i in range(17, 23):
    AllData[i]["planeDelays"] = AllData[i]["planeDelays"].T
sessId, nId = GetNeuronDetails(AllData)
tups = np.vstack((sessId, nId)).T
tups = list(tuple(map(tuple, tups)))
# hack for new dataset


#%% Panel B
plt.close("all")
a = GetChoicePerAnimal(AllData, iteration=0, plot=1)
#%% Panel C
plt.close("all")
b, c = GetReactionTime(AllData, plot=1)

#%% Panel E
plt.close("all")
n = 13
tups = np.vstack((sessId, nId)).T
session_r = tups[:, 0]
r_sess = r[:, session_r == n]
p_sess = p[:, session_r == n]
d = AllData[n]
sig = d["calTrace"]
stimSt = d["stimT"][:, 0]
timeTrace = d["calTimes"]
cl = d["contrastLeft"]
cr = d["contrastRight"]
feedbackTimes = d["feedbackTimes"]
feedbackType = d["feedbackType"][:, 0]
wheelTime = d["wheelTimes"]
wheelVelocity = d["wheelVelocity"]
pupilTime = d["pupilTimes"]
pupil = d["pupil"]
beepTimes = d["goCueTimes"][:, 0]
feedBackPeriod = d["posFeedbackPeriod"]

# Lick
lickraw = d["lick_raw"][:, 0]
lickTimes = d["wheelTimes"]
lcks = detectLicks(lickraw)
lcks = lickTimes[lcks]
lcks = cleanLickBursts(lcks, 0.3)

# Take only licks that are not reward related - not need for general view
# removeLcks = []
# for lckI in range(len(lcks)):
#     lck = lcks[lckI]
#     timeFromReward = lck - feedbackTimes
#     timeFromReward = timeFromReward[timeFromReward > 0]
#     if len(timeFromReward) == 0:
#         continue
#     shortestTimeFromReward = np.min(timeFromReward)
#     # within reward time
#     if (shortestTimeFromReward) < np.max(feedBackPeriod) + 1:
#         removeLcks.append(lckI)
# lcks = np.delete(lcks, removeLcks)
####################################################
sigs, sortInd = PrintPopulationResponsiveness(
    {1: d}, sortBy=r_sess, addLines=False, returnIndex=True
)
plt.close("all")
visInd = np.where(r_sess[0, :])[0]
moveInd = np.where((r_sess[3, :]) | (r_sess[4, :]))[0]
feedbacInd = np.where((r_sess[5, :]) | (r_sess[6, :]))[0]
lickInd = np.where((r_sess[-1, :]))[0]

f, ax = plt.subplots(1 + 5, 1, sharex=True)
ax[0].plot(pupilTime, pupil, "b")
ax[0].set_ylabel("pupil")
ax[1].vlines(stimSt, 0, 1 * (cl - cr), "k")
ax[1].set_ylabel("contrast")
ax[2].vlines(feedbackTimes[feedbackType == 1, 0], 0, 1, "g")
ax[2].vlines(feedbackTimes[feedbackType == 0, 0], 0, 1, "r")
ax[2].set_ylabel("feedback")
ax[3].vlines(lcks, 0, 1, "k")
ax[3].set_ylabel("lick")
ax[4].plot(wheelTime, wheelVelocity, "k")
ax[4].set_ylabel("wheel")
# ax[5].plot(timeTrace, sig[:, visInd[0]])
# ax[6].plot(timeTrace, sig[:, moveInd[0]])
# ax[7].plot(timeTrace, sig[:, feedbacInd[0]])
# ax[8].plot(timeTrace, sig[:, lickInd[0]])
ax[5].imshow(
    sp.stats.zscore(sig[:, sortInd].T, nan_policy="omit", axis=1),
    extent=(timeTrace[0, 0], timeTrace[-1, 0], 0, sig.shape[1]),
    aspect="auto",
    cmap="binary",
)

ax[5].set_xlabel("Time (s)")

# for i in range(5, 9):
#     None
#     ax[i].set_ylim(-1, 12)
ax[0].set_xlim(1905, 2250)

#%% Panel F
plt.close("all")
PrintSignleResponse(AllData, 10, 29, False)
PrintSignleResponse(AllData, 10, 30, False)
PrintSignleResponse(AllData, 10, 70, False)

#%% Panel G

plt.close("all")
nums = []

overlaps = []
# all responses
fall, axAll = plt.subplots(1)
fall.suptitle("All sessions")
axAll.set_xlim(-120, 120)
axAll.set_ylim(-120, 120)
axAll.set_aspect("equal")
# stim

for i in range(1, 17):

    d = AllData[i]
    sig = d["calTrace"]
    if not ("rfParams" in d.keys()):
        nums.append((sig.shape[1], 0, 0))
        overlaps.append(np.ones(sig.shape[1]) * np.nan)
        continue

    # normalise sig
    # sig/=np.nanmax(sig,axis = 0)

    rfParams = d["rfParams"]
    rfEV = d["rfEV"]
    rfPeaks = d["rfPeaks"]
    rfEdges = d["rfEdges"][0, :]
    rfParams[(rfEV[:, 0] <= 0.015) | (rfPeaks[:, 0] <= 3.5), :] = np.nan
    rfAz = np.abs(rfParams[:, 1])
    rfAlt = np.abs(rfParams[:, 3])
    rfSig = (rfParams[:, 2] + rfParams[:, 4]) / 2
    rfSig *= 1.5
    rfArea = np.pi * rfSig**2
    stimAlt = d["stimAlt"][0, 0]
    stimAz = d["stimDist"][0, 0]
    stimSig = d["stimSigma"][0, 0]

    nums.append(
        (
            sig.shape[1],
            len(rfSig),
            rfParams.shape[0]
            - np.sum((rfEV[:, 0] <= 0.015) | (rfPeaks[:, 0] <= 3.5)),
        )
    )
    dist = np.abs(np.sqrt((rfAz - stimAz) ** 2 + (rfAlt - stimAlt) ** 2))

    overlap = (
        rfSig**2
        * (
            np.abs(
                np.arccos(
                    (dist**2 + rfSig**2 - stimSig**2)
                    / (2 * dist * rfSig)
                    + 0j
                )
            )
        )
        + stimSig**2
        * np.abs(
            np.arccos(
                (dist**2 + stimSig**2 - rfSig**2) / (2 * dist * stimSig)
                + 0j
            )
        )
        - 0.5
        * np.sqrt(
            (-dist + stimSig + rfSig)
            * (dist + stimSig - rfSig)
            * (dist - stimSig + rfSig)
            * (dist + stimSig + rfSig)
        )
    )
    percentOverlap = overlap / rfArea

    percentOverlap[dist >= stimSig + rfSig] = 0

    if sig.shape[1] > len(rfSig):
        poverlap = np.ones(sig.shape[1]) * np.nan
        corrs = np.zeros((sig.shape[1], len(rfSig)))
        sig2 = np.load("D:\\_ss_2pCalcium.dff.npy")
        for j in range(sig.shape[1]):
            for k in range(len(rfSig)):
                sig[np.isnan(sig)] = 0
                sig2[np.isnan(sig2)] = 0
                corrs[j, k], _ = sp.stats.pearsonr(sig[:, j], sig[:, k])
        actualNeurons = np.nanargmax(corrs, axis=0)
        rfParams = rfParams[actualNeurons, :]

        poverlap[actualNeurons] = percentOverlap
        percentOverlap = poverlap

    overlaps.append(percentOverlap)

    circInds = np.where(~np.all(np.isnan(rfParams), axis=1))[0]
    for ci in circInds:
        crc = Circle(
            [rfAz[ci] - stimAz, rfAlt[ci] - stimAlt],
            rfSig[ci],
            facecolor=None,
            edgecolor="black",
            fill=False,
        )
        axAll.add_patch(crc)
crc = Circle(
    [0, 0],
    stimSig,
    facecolor="Red",
    edgecolor="black",
    fill=True,
)
axAll.add_patch(crc)
axAll.set_xlim(-80, 40)
axAll.set_ylim(-50, 50)

#%% Panels H
plt.close("all")
overlapsF = np.hstack(overlaps)
overlapsF[overlapsF == 0] = -0.1
tups = np.vstack((sessId, nId)).T
vresp = r[0, :]
vresp = vresp[sessId <= 16]
numsS = np.vstack(nums)

bins = np.arange(-0.1, 1.1, 0.1)

f, ax = plt.subplots(1)
# sns.histplot(overlapsF, bins=bins, color="black", stat='percent', palette=['black'])
ax.hist(overlapsF, bins=bins, color="black", density=100)
ax.set_xlabel("Overlap")
ax.set_ylabel("Ratio 1/10 percent")
#%% Panels I
plt.close("all")
resp = np.zeros(len(bins) - 1)
counts = np.zeros(len(bins) - 1)
for bi, b in enumerate(bins[:-1]):
    inds = np.where((overlapsF >= b) & (overlapsF <= bins[bi + 1]))[0]
    counts[bi] = len(inds)
    resp[bi] = np.sum(vresp[inds])
f, ax = plt.subplots(1)
ax.bar(bins[:-1] + 0.1, resp / counts, color="black", width=0.1)
ax.set_xlabel("Overlap")
ax.set_ylabel("Percent Responsive")
