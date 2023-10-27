# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 11:22:48 2023

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
from matplotlib.colors import ListedColormap
from matplotlib import cm


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
# d = list(tuple(map(tuple,tups)))

with open(saveDir + "responsiveness.pickle", "rb") as f:
    res = pickle.load(f)
resp = res["Resp"]
p = res["pVals"]
z = res["zScores"]
r = p <= 0.05 / 8


def GetSscDepth(fits, AllSubjects):

    dpa = {
        "SS087": -298,
        "SS088": -330,
        "SS089": -338,
        "SS091": -314,
        "SS092": -310,
        "SS093": -277,
    }

    animalId = np.vstack(fits.Id)[:, 0]
    animals = np.vstack(AllSubjects.values())[animalId]
    super_border = [dpa[x] for x in np.squeeze(animals)]
    return super_border


#%% Panel C
tups = np.vstack((sessId, pId, nId)).T


modelsBase = GetOneFunctionVarianceDifferentModels(
    dataDir="D:\\Figures\\OneFunction_ephys_raw\\",
    cutoff=0.0001,
    shuffles=True,
    resp=tups,
)


f1 = np.vstack(modelsBase[modelsBase["bestFit"] == "F"]["Id"])
fits = modelsBase[modelsBase["bestFit"] == "F"].reset_index(drop=True)
dspf = GetDepthFromList(AllData, fits, "D:\\Figures\\OneFunction_ephys_raw\\")
dspf -= GetSscDepth(fits, AllSubjects)


n = 46


curr = f1[n, :]


tups = np.vstack((sessId, pId, nId)).T


d = AllData[curr[0]]


movOnset = d["wheelMoveIntervals"][:, 0]
choice = d["wheelDisplacement"][:, 0]
choice[choice > 0] = 1
choice[choice < 0] = -1

stimStarts = d["stimT"][:, 0]
stimEnds = d["stimT"][:, 1]


p = d["probes"][curr[1] + 1]
imagingSide = p["ImagingSide"]


scDepth = p["scDepth"][0]

spikes = MakeSpikeInfoMatrix(p, [scDepth - 1000, scDepth])
uniqueClust = np.unique(spikes[:, 1])
unitId = uniqueClust[curr[-1]]

if imagingSide == 1:
    contra = d["contrastLeft"][:, 0]
    ipsi = d["contrastRight"][:, 0]
    choice = -choice
else:
    ipsi = d["contrastLeft"][:, 0]
    contra = d["contrastRight"][:, 0]
contraU = np.unique(contra)
ch = d["choice"]
beepStarts = d["goCueTimes"]
feedbackStarts = d["feedbackTimes"]
feedBackPeriod = d["posFeedbackPeriod"]
feedbackType = d["feedbackType"][:, 0]
prevfeedback = np.append([0], feedbackType[:-1])

window = np.array([-0.5 - 0.5, 0.5 + 0.5])


stimA, t = GetRaster(spikes.copy(), stimStarts, window, dt=1 / 1000)


lenContraN = np.zeros(len(contraU) + 1)
lenContraP = np.zeros(len(contraU) + 1)
maxCount = np.zeros(len(contraU) + 1)
# contraU = contraU[contraU>0]
for ci, c in enumerate(contraU):
    IndN = np.where((contra == c) & (prevfeedback == 0))[0]
    IndP = np.where((contra == c) & (prevfeedback == 1))[0]
    lenContraN[ci + 1] = len(IndN)
    lenContraP[ci + 1] = len(IndP)
    lens = [len(IndN), len(IndP)]
    maxCount[ci + 1] = lens[np.argmax(lens)]


f, ax = plt.subplots(2, 2, sharex=True)
stimA_ = stimA[:, :, curr[2]].copy()
lenContra = np.zeros(len(contraU) + 1)
lenContraR = np.zeros(len(contraU) + 1)
lastI = 0
colors = np.linspace(0.8, 0, len(contraU))
# print neg
for ci, c in enumerate(contraU):
    Ind = np.where((contra == c) & (prevfeedback == 0))[0]
    lenContra[ci + 1] = len(Ind)
    psth = GetPSTH(stimA_[:, Ind], w=30, halfGauss=True, flat=False)
    m = np.nanmean(psth, 1)
    sd = np.nanstd(psth, 1) / np.sqrt(psth.shape[1])
    ax[1, 0].plot(t, m, color=(colors[ci], colors[ci], colors[ci]))
    ax[1, 0].fill_between(
        t,
        m - sd,
        m + sd,
        color=(colors[ci], colors[ci], colors[ci]),
        alpha=0.2,
    )
    contrastStartPoint = np.cumsum(maxCount)[ci]
    for i in range(len(Ind)):
        spLocal = stimA_[:, Ind[i]]
        ax[0, 0].vlines(
            t[spLocal],
            contrastStartPoint + i,
            contrastStartPoint + i + 1,
            color=(colors[ci], colors[ci], colors[ci]),
        )

    lastI += i
    lenContraR[ci + 1] = lenContra[ci + 1].copy()
    lenContra[ci + 1] += lastI


# print pos
lenContra2 = np.zeros(len(contraU) + 1)
lastI = 0
for ci, c in enumerate(contraU):
    Ind = np.where((contra == c) & (prevfeedback == 1))[0]
    lenContra2[ci + 1] = len(Ind)
    psth = GetPSTH(stimA_[:, Ind], w=30, halfGauss=True, flat=False)
    m = np.nanmean(psth, 1)
    sd = np.nanstd(psth, 1) / np.sqrt(psth.shape[1])
    ax[1, 1].plot(t, m, color=(colors[ci], colors[ci], colors[ci]))
    ax[1, 1].fill_between(
        t,
        m - sd,
        m + sd,
        color=(colors[ci], colors[ci], colors[ci]),
        alpha=0.2,
    )
    contrastStartPoint = np.cumsum(maxCount)[ci]
    for i in range(len(Ind)):
        ax[0, 1].vlines(
            t[stimA_[:, Ind[i]]],
            contrastStartPoint + i,
            contrastStartPoint + i + 1,
            color=(colors[ci], colors[ci], colors[ci]),
        )
    lastI += i


ax[0, 0].set_xlim(-0.1, 0.5)

maxY0 = np.max(np.nanmax([np.sum(lenContra), np.sum(lenContra2)]))
maxY1 = np.max([ax[1, 0].get_ylim()[1], ax[1, 1].get_ylim()[1]])

ax[1, 0].set_ylim(0, maxY1)
ax[1, 1].set_ylim(0, maxY1)
ax[0, 1].legend(contraU, bbox_to_anchor=(1.05, 1.0), loc="upper left")

#%% Panel D
tups = np.vstack((sessId, pId, nId)).T


modelsBase = GetOneFunctionVarianceDifferentModels(
    dataDir="D:\\Figures\\OneFunction_ephys_raw\\",
    cutoff=0.0001,
    shuffles=True,
    resp=tups,
)


f1 = np.vstack(modelsBase[modelsBase["bestFit"] == "P"]["Id"])
fits = modelsBase[modelsBase["bestFit"] == "P"].reset_index(drop=True)
dspf = GetDepthFromList(AllData, fits, "D:\\Figures\\OneFunction_ephys_raw\\")
dspf -= GetSscDepth(fits, AllSubjects)
#%%

n = 15


curr = f1[n, :]


tups = np.vstack((sessId, pId, nId)).T


d = AllData[curr[0]]


movOnset = d["wheelMoveIntervals"][:, 0]
choice = d["wheelDisplacement"][:, 0]
choice[choice > 0] = 1
choice[choice < 0] = -1
# pupil_ds = AlignSignals(d['pupil'],d['pupilTimes'],d['times'])
# pudt = np.mean(np.diff(d['pupilTimes'][:,0]))
# tdt = np.mean(np.diff(d['times'])
# actTimes = actTimes[((actTimes>=d['times'][0]) & (actTimes<=d['times'][-1]))]

# pupil_ds,put = sp.signal.resample(d['pupil'],actTimes,len(d['times']))

stimStarts = d["stimT"][:, 0]
stimEnds = d["stimT"][:, 1]


p = d["probes"][curr[1] + 1]
imagingSide = p["ImagingSide"]


scDepth = p["scDepth"][0]

spikes = MakeSpikeInfoMatrix(p, [scDepth - 1000, scDepth])
uniqueClust = np.unique(spikes[:, 1])
unitId = uniqueClust[curr[-1]]

if imagingSide == 1:
    contra = d["contrastLeft"][:, 0]
    ipsi = d["contrastRight"][:, 0]
    choice = -choice
else:
    ipsi = d["contrastLeft"][:, 0]
    contra = d["contrastRight"][:, 0]
contraU = np.unique(contra)
ch = d["choice"]
beepStarts = d["goCueTimes"]
feedbackStarts = d["feedbackTimes"]
feedBackPeriod = d["posFeedbackPeriod"]
feedbackType = d["feedbackType"][:, 0]
prevfeedback = np.append([0], feedbackType[:-1])

window = np.array([-0.5 - 0.5, 0.5 + 0.5])


windowP = np.array([[-0.5, 0.5]])
pupil_ds = d["pupil"]

taskStart = np.argmin(np.abs(d["pupilTimes"] - stimStarts[0]))
taskEnd = np.argmin(np.abs(d["pupilTimes"] - stimStarts[-1]))
taskPupl = pupil_ds[taskStart:taskEnd]
# middle = np.nanmedian(pupil_ds,0)
middle = np.nanmedian(taskPupl, 0)
pu, t = AlignStim(pupil_ds, d["pupilTimes"], stimStarts, windowP)
bigTrials = pu > middle
bigTrialsSum = np.sum(bigTrials[:, :, 0], 0)
meanPupil = np.nanmean(pu[:, :, 0], 0)
largePupil = (bigTrialsSum / len(t)) > 0.5
smallPupil = (bigTrialsSum / len(t)) < 0.5

prevFeedback = largePupil


stimA, t = GetRaster(spikes.copy(), stimStarts, window, dt=1 / 1000)


# first find how much trails in each case
lenContraN = np.zeros(len(contraU) + 1)
lenContraP = np.zeros(len(contraU) + 1)
maxCount = np.zeros(len(contraU) + 1)
# contraU = contraU[contraU>0]
for ci, c in enumerate(contraU):
    IndN = np.where((contra == c) & (prevfeedback == 0))[0]
    IndP = np.where((contra == c) & (prevfeedback == 1))[0]
    lenContraN[ci + 1] = len(IndN)
    lenContraP[ci + 1] = len(IndP)
    lens = [len(IndN), len(IndP)]
    maxCount[ci + 1] = lens[np.argmax(lens)]


f, ax = plt.subplots(2, 2, sharex=True)
stimA_ = stimA[:, :, curr[2]].copy()
lenContra = np.zeros(len(contraU) + 1)
lenContraR = np.zeros(len(contraU) + 1)
lastI = 0
colors = np.linspace(0.8, 0, len(contraU))
# print neg
for ci, c in enumerate(contraU):
    Ind = np.where((contra == c) & (prevfeedback == 0))[0]
    lenContra[ci + 1] = len(Ind)
    psth = GetPSTH(stimA_[:, Ind], w=30, halfGauss=True, flat=False)
    m = np.nanmean(psth, 1)
    sd = np.nanstd(psth, 1) / np.sqrt(psth.shape[1])
    ax[1, 0].plot(t, m, color=(colors[ci], colors[ci], colors[ci]))
    ax[1, 0].fill_between(
        t,
        m - sd,
        m + sd,
        color=(colors[ci], colors[ci], colors[ci]),
        alpha=0.2,
    )
    contrastStartPoint = np.cumsum(maxCount)[ci]
    for i in range(len(Ind)):
        spLocal = stimA_[:, Ind[i]]
        ax[0, 0].vlines(
            t[spLocal],
            contrastStartPoint + i,
            contrastStartPoint + i + 1,
            color=(colors[ci], colors[ci], colors[ci]),
        )

    lastI += i
    lenContraR[ci + 1] = lenContra[ci + 1].copy()
    lenContra[ci + 1] += lastI


# print pos
lenContra2 = np.zeros(len(contraU) + 1)
lastI = 0
for ci, c in enumerate(contraU):
    Ind = np.where((contra == c) & (prevfeedback == 1))[0]
    lenContra2[ci + 1] = len(Ind)
    psth = GetPSTH(stimA_[:, Ind], w=30, halfGauss=True, flat=False)
    m = np.nanmean(psth, 1)
    sd = np.nanstd(psth, 1) / np.sqrt(psth.shape[1])
    ax[1, 1].plot(t, m, color=(colors[ci], colors[ci], colors[ci]))
    ax[1, 1].fill_between(
        t,
        m - sd,
        m + sd,
        color=(colors[ci], colors[ci], colors[ci]),
        alpha=0.2,
    )
    contrastStartPoint = np.cumsum(maxCount)[ci]
    for i in range(len(Ind)):
        ax[0, 1].vlines(
            t[stimA_[:, Ind[i]]],
            contrastStartPoint + i,
            contrastStartPoint + i + 1,
            color=(colors[ci], colors[ci], colors[ci]),
        )
    lastI += i


# * np.arange(1,stimA.shape[1]+1).T


# start with negative


ax[0, 0].set_xlim(-0.1, 0.5)
# ax[0, 0].set_ylim(0, stimA.shape[1])
maxY0 = np.max(np.nanmax([np.sum(lenContra), np.sum(lenContra2)]))
maxY1 = np.max([ax[1, 0].get_ylim()[1], ax[1, 1].get_ylim()[1]])

# ax[0,0].set_ylim(0,maxY0)
# ax[0,1].set_ylim(0,maxY0)
ax[1, 0].set_ylim(0, maxY1)
ax[1, 1].set_ylim(0, maxY1)
ax[0, 1].legend(contraU, bbox_to_anchor=(1.05, 1.0), loc="upper left")

#%% Panel E


colours = np.zeros((256, 4))
red = np.ones(256)
green = np.append(np.ones(86), np.linspace(1, 0.3, 256 - 86))  # np.append(

# np.linspace(0.5, 0, 128))
blue = np.append(np.linspace(1, 0.1, 86), np.ones(256 - 86) * 0)  # np.append(

alphas = np.ones_like(red)
colours[:, 0] = red
colours[: len(green), 1] = green
colours[: len(blue), 2] = blue
colours[:, -1] = alphas
newcmp = ListedColormap(colours)


plt.close("all")
tups = np.vstack((sessId, pId, nId)).T
tups = tups[r[0, :] == 1, :]
window = np.array([[-0.2, 1]])[0]
modelsBase = GetOneFunctionVarianceDifferentModels(
    dataDir="D:\\Figures\\OneFunction_ephys_raw\\",
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


prevPosResp = []
prevNegResp = []
t_list = []
for i in range(1, len(AllData) + 1):
    if not (i in f1[:, 0]):
        continue
    print(f"Neuron {str(i)}")
    relProbes = np.unique(f1[f1[:, 0] == i, 1])

    for pi in relProbes:
        relNeurons = f1[(f1[:, 0] == i) & (f1[:, 1] == pi), -1]
        p = AllData[i]["probes"][pi + 1]

        imagingSide = p["ImagingSide"]

        scDepth = p["scDepth"][0]
        spikes = MakeSpikeInfoMatrix(p, [scDepth - 1000, scDepth])
        uniqueClust = np.unique(spikes[:, 1])
        # spikesCell = spikes[spikes[:, 1] == uniqueClust[relNeurons],:]

        stimStarts = AllData[i]["stimT"][:, 0]

        choice = AllData[i]["choice"]
        feedback = AllData[i]["feedbackType"]
        prevFb = np.append(0, feedback[:-1])

        if imagingSide == 1:
            contra = AllData[i]["contrastLeft"][:, 0]
            ipsi = AllData[i]["contrastRight"][:, 0]
            choice = -choice
        else:
            ipsi = AllData[i]["contrastLeft"][:, 0]
            contra = AllData[i]["contrastRight"][:, 0]

        relevantTrial = np.where(contra > 0)[0]
        choice = choice[relevantTrial]
        feedback = feedback[relevantTrial]
        stimStarts = stimStarts[relevantTrial]
        prevFb = prevFb[relevantTrial]
        contra = contra[relevantTrial]
        stimA, t = GetRaster(spikes.copy(), stimStarts, window, dt=1 / 1000)

        stimA = stimA[:, :, relNeurons].astype(float)
        for nn in range(stimA.shape[-1]):
            stimA[:, :, nn] = GetPSTH(
                stimA[:, :, nn], w=30, halfGauss=True, flat=False
            )

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
                stimA[:, (contra == np.nanmax(contra)) & (prevFb == 1), :],
                1,
            )
        )
        prevNegResp.append(
            np.nanmean(
                stimA[:, (contra == np.nanmax(contra)) & (prevFb == 0), :],
                1,
            )
        )
        t_list.append(t)

slowestT = t_list[0]
prevPosResp_ = prevPosResp.copy()
prevNegResp_ = prevNegResp.copy()
prevPosResp = np.hstack(prevPosResp)
prevNegResp = np.hstack(prevNegResp)

ratio = prevPosResp / prevNegResp

meanRatio = np.nanmean(ratio[(slowestT >= 0) & (slowestT <= 0.5)], axis=0)

df = (t[1] - t[0]) / 2

colours = np.zeros((256, 4))
red = np.ones(256)
green = np.append(
    np.ones(128), np.linspace(1, 0.3, 128)
)  # np.append(np.ones(86), np.linspace(1, 0.3, 256 - 86))  # np.append(
#     np.ones(64), np.linspace(1, 0.3, 256 - 64)
# )  # np.append(np.ones(128), np.linspace(1, 0.3, 128))##

# np.linspace(0.5, 0, 128))
blue = np.append(
    np.linspace(1, 0.1, 128), np.ones(128) * 0
)  # np.append(np.linspace(1, 0.1, 86), np.ones(256 - 86) * 0)  # np.append(

alphas = np.ones_like(red)
colours[:, 0] = red
colours[: len(green), 1] = green
colours[: len(blue), 2] = blue
colours[:, -1] = alphas
newcmp = ListedColormap(colours)


###for 3
colours = np.zeros((256, 4))
red = np.ones(256)
green = np.append(np.ones(86), np.linspace(1, 0.3, 256 - 86))  # np.append(
#     np.ones(64), np.linspace(1, 0.3, 256 - 64)
# )  # np.append(np.ones(128), np.linspace(1, 0.3, 128))##

# np.linspace(0.5, 0, 128))
blue = np.append(np.linspace(1, 0.1, 86), np.ones(256 - 86) * 0)  # np.append(

alphas = np.ones_like(red)
colours[:, 0] = red
colours[: len(green), 1] = green
colours[: len(blue), 2] = blue
colours[:, -1] = alphas
newcmp = ListedColormap(colours)

### second option
negMax = np.nanmax(prevNegResp[(slowestT >= 0) & (slowestT <= 0.2)], 0)
negNorm = prevNegResp / negMax
posNorm = prevPosResp / negMax


meanRatio = np.nanmax(posNorm[(slowestT >= 0) & (slowestT <= 0.2)], axis=0)


f, ax = plt.subplots(1, 2, sharex=True)
im = ax[0].imshow(
    np.flip(negNorm[:, np.argsort(meanRatio)].T, axis=0),
    cmap=newcmp,  # "YlOrRd",
    vmin=0,
    vmax=2,
    extent=(slowestT[0] + df, slowestT[-1] + df, 0, ratio.shape[1]),
    aspect="auto",
)
im = ax[1].imshow(
    np.flip(posNorm[:, np.argsort(meanRatio)].T, axis=0),
    cmap=newcmp,  # "YlOrRd",
    vmin=0,
    vmax=2,
    extent=(slowestT[0] + df, slowestT[-1] + df, 0, ratio.shape[1]),
    aspect="auto",
)
ax[1].set_xlim(-0.25, 1)
f.colorbar(im)

#%% Panel F
plt.close("all")
tups = np.vstack((sessId, pId, nId)).T
tups = tups[r[0, :] == 1, :]
window = np.array([[-0.2, 1]])[0]
modelsBase = GetOneFunctionVarianceDifferentModels(
    dataDir="D:\\Figures\\OneFunction_ephys_raw\\",
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


prevPosResp = []
prevNegResp = []
t_list = []
for i in range(1, len(AllData) + 1):
    if not (i in f1[:, 0]):
        continue
    print(f"Neuron {str(i)}")
    relProbes = np.unique(f1[f1[:, 0] == i, 1])

    for pi in relProbes:
        relNeurons = f1[(f1[:, 0] == i) & (f1[:, 1] == pi), -1]
        p = AllData[i]["probes"][pi + 1]

        imagingSide = p["ImagingSide"]

        scDepth = p["scDepth"][0]
        spikes = MakeSpikeInfoMatrix(p, [scDepth - 1000, scDepth])
        uniqueClust = np.unique(spikes[:, 1])
        # spikesCell = spikes[spikes[:, 1] == uniqueClust[relNeurons],:]

        stimStarts = AllData[i]["stimT"][:, 0]

        choice = AllData[i]["choice"]
        feedback = AllData[i]["feedbackType"]
        prevFb = np.append(0, feedback[:-1])

        if imagingSide == 1:
            contra = AllData[i]["contrastLeft"][:, 0]
            ipsi = AllData[i]["contrastRight"][:, 0]
            choice = -choice
        else:
            ipsi = AllData[i]["contrastLeft"][:, 0]
            contra = AllData[i]["contrastRight"][:, 0]

        middle = np.nanmedian(
            AllData[i]["pupil"][AllData[i]["pupilTimes"] < stimStarts[-1]], 0
        )
        pu, tp = AlignStim(
            AllData[i]["pupil"],
            AllData[i]["pupilTimes"],
            stimStarts.reshape(-1, 1),
            window.reshape(1, -1),
        )
        bigTrials = pu > middle
        bigTrialsSum = np.sum(bigTrials[:, :, 0], 0)
        meanPupil = np.nanmean(pu[:, :, 0], 0)
        largePupil = (bigTrialsSum / len(tp)) > 0.5
        smallPupil = (bigTrialsSum / len(tp)) < 0.5

        relevantTrial = np.where(contra > 0)[0]
        choice = choice[relevantTrial]
        feedback = feedback[relevantTrial]
        stimStarts = stimStarts[relevantTrial]
        prevFb = prevFb[relevantTrial]
        contra = contra[relevantTrial]
        largePupil = largePupil[relevantTrial]
        stimA, t = GetRaster(spikes.copy(), stimStarts, window, dt=1 / 1000)

        stimA = stimA[:, :, relNeurons].astype(float)
        for nn in range(stimA.shape[-1]):
            stimA[:, :, nn] = GetPSTH(
                stimA[:, :, nn], w=30, halfGauss=True, flat=False
            )

        # cr = cr + stimA.shape[-1]
        pre = np.nanmean(stimA[t <= 0, :, :], 0)
        post = np.nanmean(stimA[(t > 0.2) & (t <= 1), :, :], 0)

        # if i == 22:
        #     print("Ya")
        flipInds = np.where(np.nanmean(post, 0) < np.nanmean(pre, 0))[0]
        if len(flipInds) > 0:
            stimA[:, :, flipInds] = -stimA[:, :, flipInds]

        # stimA = stimA[:, :, :]
        # -np.nanmean(
        #     stimA[(t <= 0) & (t >= -0.1), :, :][:, :, :],
        #     0,
        # )
        stimA -= np.nanmean(
            stimA[(t <= 0), :, :],
            0,
        )

        prevPosResp.append(
            np.nanmean(
                stimA[:, (contra == np.nanmax(contra)) & (prevFb == 1), :],
                1,
            )
        )
        prevNegResp.append(
            np.nanmean(
                stimA[:, (contra == np.nanmax(contra)) & (prevFb == 0), :],
                1,
            )
        )
        t_list.append(t)

slowestT = t_list[0]
prevPosResp_ = prevPosResp.copy()
prevNegResp_ = prevNegResp.copy()
prevPosResp = np.hstack(prevPosResp)
prevNegResp = np.hstack(prevNegResp)

ratio = prevPosResp / prevNegResp

meanRatio = np.nanmean(ratio[(slowestT >= 0) & (slowestT <= 0.05)], axis=0)


#### for 2
colours = np.zeros((256, 4))
red = np.ones(256)
green = np.append(
    np.ones(128), np.linspace(1, 0.3, 128)
)  # np.append(np.ones(86), np.linspace(1, 0.3, 256 - 86))  # np.append(


# np.linspace(0.5, 0, 128))
blue = np.append(
    np.linspace(1, 0.1, 128), np.ones(128) * 0
)  # np.append(np.linspace(1, 0.1, 86), np.ones(256 - 86) * 0)  # np.append(

alphas = np.ones_like(red)
colours[:, 0] = red
colours[: len(green), 1] = green
colours[: len(blue), 2] = blue
colours[:, -1] = alphas
newcmp = ListedColormap(colours)


negMax = np.nanmax(prevNegResp[(slowestT >= 0) & (slowestT <= 0.2)], 0)
negNorm = prevNegResp / negMax
posNorm = prevPosResp / negMax

meanRatio = np.nanmax(posNorm[(slowestT >= 0) & (slowestT <= 0.2)], axis=0)


f, ax = plt.subplots(1, 2, sharex=True)
im = ax[0].imshow(
    np.flip(negNorm[:, np.argsort(meanRatio)].T, axis=0),
    cmap=newcmp,
    vmin=0,
    vmax=2,
    extent=(slowestT[0] + df, slowestT[-1] + df, 0, ratio.shape[1]),
    aspect="auto",
)
im = ax[1].imshow(
    np.flip(posNorm[:, np.argsort(meanRatio)].T, axis=0),
    cmap=newcmp,
    vmin=0,
    vmax=2,
    extent=(slowestT[0] + df, slowestT[-1] + df, 0, ratio.shape[1]),
    aspect="auto",
)

ax[1].set_xlim(-0.25, 1)
f.colorbar(im)


#%% Panels G-J

nullProps1 = GetNullDistribution(allFit, 4)
nullProps2 = GetNullDistribution(allFit, 1)


def CreateNullIntervalPlot(ax, dist):
    ys = []
    f, axt = plt.subplots(1)
    # sns.histplot(data= dist.T,legend=False,common_norm=False,cumulative=True, stat='proportion',element='poly',fill=False,binrange=(-2,2),binwidth=0.3,ax=axt)
    sns.kdeplot(
        data=dist,
        legend=False,
        common_norm=False,
        common_grid=True,
        cumulative=True,
        ax=axt,
        clip=(-2, 2),
        bw_adjust=0.1,
        cut=0,
    )
    for i in range(len(axt.lines)):
        line = axt.lines[i]
        x, y = line.get_data()
        y /= np.nanmax(y)
        ys.append(y)
        plt.close(f)
    ys = np.vstack(ys)

    # ax.plot(x,np.nanmean(ys,0),'grey')
    ax.fill_between(
        x,
        np.nanpercentile(ys, 2.5, axis=0),
        np.nanpercentile(ys, 97.5, axis=0),
        color="grey",
        alpha=0.4,
    )
    return None


tups = np.vstack((sessId, pId, nId)).T
tups = tups[r[0, :] == 1, :]
modelsBase = GetOneFunctionVarianceDifferentModels(
    dataDir="D:\\Figures\\OneFunction_ephys_raw\\",
    cutoff=0.01,
    shuffles=True,
    resp=tups,
)

plt.close("all")

depthList = -GetDepthFromList(
    AllData, modelsBase, "D:\\Figures\\OneFunction_ephys_raw\\"
)
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
allFit = np.array(modelsBase["fit"])

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


bins = np.arange(-1, 1.1, 0.03)
ymax = 80
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

ff, axt = plt.subplots(1)
sns.kdeplot(
    data=[valsPos],
    palette=["black"],
    ax=axt,
    legend=False,
    common_norm=False,
    cumulative=True,
    clip=(-2, 2),
    bw_adjust=0.1,
    cut=0,
)
line = axt.lines[0]
x, y = line.get_data()
y /= np.max(y)
plt.close(ff)
plt.plot(x, y, "k")
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

ff, axt = plt.subplots(1)
sns.kdeplot(
    data=[valsPos],
    palette=["black"],
    ax=axt,
    legend=False,
    common_norm=False,
    cumulative=True,
    clip=(-2, 2),
    bw_adjust=0.1,
    cut=0,
)
line = axt.lines[0]
x, y = line.get_data()
y /= np.max(y)
plt.close(ff)
plt.plot(x, y, "k")

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

#%% Panel K


def GetSscDepth(fits, AllSubjects):

    dpa = {
        "SS087": -298,
        "SS088": -330,
        "SS089": -338,
        "SS091": -314,
        "SS092": -310,
        "SS093": -277,
    }

    animalId = np.vstack(fits.Id)[:, 0]
    animals = np.vstack(AllSubjects.values())[animalId]
    super_border = [dpa[x] for x in np.squeeze(animals)]
    return super_border


plt.close("all")
tups = np.vstack((sessId, pId, nId)).T
tups = tups[r[0, :] == 1, :]
modelsBase = GetOneFunctionVarianceDifferentModels(
    dataDir="D:\\Figures\\OneFunction_ephys_raw\\",
    flat=False,
    cutoff=0.01,
    shuffles=True,
    resp=tups,
)
fits = modelsBase.reset_index(drop=True)
ds_general = GetDepthFromList(
    AllData, fits, "D:\\Figures\\OneFunction_ephys_raw\\"
)
ds_general -= GetSscDepth(fits, AllSubjects)

print("P")
fits = modelsBase[modelsBase["bestFit"] == "P"].reset_index(drop=True)
dsp = GetDepthFromList(AllData, fits, "D:\\Figures\\OneFunction_ephys_raw\\")
dsp -= GetSscDepth(fits, AllSubjects)
print("PF")
fits = modelsBase[modelsBase["bestFit"] == "PF"].reset_index(drop=True)
dspf = GetDepthFromList(AllData, fits, "D:\\Figures\\OneFunction_ephys_raw\\")
dspf -= GetSscDepth(fits, AllSubjects)
print("G")
fits = modelsBase[modelsBase["bestFit"] == "G"].reset_index(drop=True)
dsg = GetDepthFromList(AllData, fits, "D:\\Figures\\OneFunction_ephys_raw\\")
dsg -= GetSscDepth(fits, AllSubjects)
print("F")
fits = modelsBase[modelsBase["bestFit"] == "F"].reset_index(drop=True)
dsf = GetDepthFromList(AllData, fits, "D:\\Figures\\OneFunction_ephys_raw\\")
dsf -= GetSscDepth(fits, AllSubjects)
print("None")
fits = modelsBase[modelsBase["bestFit"] == "None"].reset_index(drop=True)
dsn = GetDepthFromList(AllData, fits, "D:\\Figures\\OneFunction_ephys_raw\\")
dsn -= GetSscDepth(fits, AllSubjects)

fits = modelsBase[modelsBase["bestFit"] == "C"].reset_index(drop=True)
dsc = GetDepthFromList(AllData, fits, "D:\\Figures\\OneFunction_ephys_raw\\")
dsc -= GetSscDepth(fits, AllSubjects)

# all
fits = modelsBase.reset_index(drop=True)
dsall = GetDepthFromList(AllData, fits, "D:\\Figures\\OneFunction_ephys_raw\\")
dsall -= GetSscDepth(fits, AllSubjects)

dsf = np.append(dsf, dspf)
dsp = np.append(dsp, dspf)

dsph, _ = np.histogram(dsp, range(-1000, 0, 333))
dsfh, _ = np.histogram(dsf, range(-1000, 0, 333))
dsgh, _ = np.histogram(dsg, range(-1000, 0, 333))
dsch, _ = np.histogram(dsc, range(-1000, 0, 333))

f, ax = plt.subplots(1)
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
sns.violinplot(data=[dsn, dsf, dsp, dsg, dsc], ax=ax, color="grey")
sns.swarmplot(
    data=[dsn, dsf, dsp, dsg, dsc], color="white", edgecolor="gray", ax=ax
)
ax.set_xticks(range(0, 5))
ax.set_xticklabels(
    ["None", "Previous Feedback", "Pupil", "Go", "Current Feedback"]
)
ax.set_ylabel("Depth relative to sSC surgace (mm)")
ax.hlines(0, -100, 100, "k", linestyles="--")

# bootstap
shuffle_n = 500
g1 = dsg
g2 = dsn
g_all = np.append(g1, g2)
g_tested = np.zeros((shuffle_n, len(g1)))

for s in range(shuffle_n):
    g_all = np.append(g1, g2)
    g1_test = np.random.choice(g_all, len(g1))
    g_tested[s, :] = g1_test

#%% Panel M
tups = np.vstack((sessId, pId, nId)).T
tups = tups[r[0, :] == 1, :]
modelsBase = GetOneFunctionVarianceDifferentModels(
    dataDir="D:\\Figures\\OneFunction_ephys_raw\\",
    cutoff=0.01,
    shuffles=True,
    resp=tups,
)
f1 = modelsBase[modelsBase["bestFit"] == "F"].append(
    modelsBase[modelsBase["bestFit"] == "PF"]
)

latencies = np.zeros(len(f1))
maxDiffPoint = np.zeros(len(f1))
peakResp = np.zeros(len(f1))
posNeg = np.zeros(len(f1))
pvalsList = []
diffsList = []
window = np.array([[-0.2, 1]])[0]
for n in range(len(f1)):
    cellInd = np.where(
        (tups[:, 0] == f1.iloc[n]["Id"][0])
        & (tups[:, 1] == f1.iloc[n]["Id"][1])
        & (tups[:, 2] == f1.iloc[n]["Id"][2])
    )
    cellData = tups[cellInd, :][0][0]
    p = AllData[cellData[0]]["probes"][cellData[1] + 1]

    imagingSide = p["ImagingSide"]

    scDepth = p["scDepth"][0]
    spikes = MakeSpikeInfoMatrix(p, [scDepth - 1000, scDepth])
    uniqueClust = np.unique(spikes[:, 1])
    spikesCell = spikes[spikes[:, 1] == uniqueClust[cellData[2]]]

    stimStarts = AllData[cellData[0]]["stimT"][:, 0]

    choice = AllData[cellData[0]]["choice"]
    feedback = AllData[cellData[0]]["feedbackType"]
    prevFb = np.append(0, feedback[:-1])

    if imagingSide == 1:
        contra = AllData[cellData[0]]["contrastLeft"][:, 0]
        ipsi = AllData[cellData[0]]["contrastRight"][:, 0]
        choice = -choice
    else:
        ipsi = AllData[cellData[0]]["contrastLeft"][:, 0]
        contra = AllData[cellData[0]]["contrastRight"][:, 0]

    relevantTrial = np.where(contra > 0)[0]
    choice = choice[relevantTrial]
    feedback = feedback[relevantTrial]
    stimStarts = stimStarts[relevantTrial]
    prevFb = prevFb[relevantTrial]

    stimA, t = GetRaster(spikesCell.copy(), stimStarts, window, dt=1 / 1000)
    stimA = GetPSTH(stimA[:, :, 0], w=5, halfGauss=True, flat=False)
    bl = np.nanmean(stimA[(t < 0), :], 0)
    stimP = stimA[:, prevFb == 1] - bl[prevFb == 1]
    stimN = stimA[:, prevFb == 0] - bl[prevFb == 0]

    # stimP = GetPSTH(stimP[:, :], w=30, halfGauss=True, flat=False)
    # stimN = GetPSTH(stimN[:, :], w=30, halfGauss=True, flat=False)
    # stimG = np.nanmean(
    #     GetPSTH(stimA[:, :, 0], w=30, halfGauss=True, flat=False), 1
    # )
    stimG = np.nanmean(stimA[:, :], 1)
    # divide time in bins
    binsSize = 0.01
    maxBin = 0.2
    nBins = int(np.floor(maxBin / binsSize))
    binEnds = np.linspace(0, maxBin, nBins)

    pVals = np.zeros_like(binEnds)[:-1]
    diffs = np.zeros_like(binEnds)[:-1]

    ### make a two way ANOVA
    # bins = []
    # types = []
    # avgs = []

    shuffleDist = []
    actualDiff = []
    # # make shuffle
    # for i in range(500):
    #     choiceInds = np.random.permutation(stimA.shape[1])
    #     sp_ = stimA[:, choiceInds[: stimP.shape[1]]]
    #     sn_ = stimA[:, choiceInds[stimP.shape[1] :]]
    #     shuffleDist.append(np.nanmean(sp_, 1) - np.nanmean(sn_, 1))

    ###############
    for b in range(len(binEnds) - 1):
        spp = np.nanmean(stimP[(t > binEnds[b]) & (t <= binEnds[b + 1]), :], 0)
        spn = np.nanmean(stimN[(t > binEnds[b]) & (t <= binEnds[b + 1]), :], 0)
        # spp = np.nansum(
        #     stimP[(t > binEnds[b]) & (t <= binEnds[b + 1]), :],
        #     0
        #     # - np.nanmean(stimP[(t > -0.2) & (t < 0), :]),
        #     # 0,
        # )
        # spn = np.nanmean(
        #     stimN[(t > binEnds[b]) & (t <= binEnds[b + 1]), :],
        #     0
        #     # - np.nanmean(stimN[(t > -0.2) & (t < 0), :]),
        #     # 0,
        # )
        # rand = np.random.choice(
        #     np.append(spp, spn), size=(len(spp) + len(spn), 500), replace=True
        # )
        # shuffleDist.append(
        #     np.abs(
        #         np.nanmean(rand[: len(spp), :], 0)
        #         - np.nanmean(rand[len(spp) :, :], 0)
        #     )
        # )
        # actualDiff.append(np.abs(np.nanmean(spp) - np.nanmean(spn)))
        # bins.append(np.ones(len(spp)+len(spn))*b)
        # types.append(np.append(np.ones(len(spp)),np.zeros(len(spn))))
        # avgs.append(np.append(spp,spn))

        # tval, p = sp.stats.ttest_ind(
        #     spp, spn, nan_policy="omit", equal_var=False,permutations=500
        # )
        _, p = sp.stats.mannwhitneyu(spp, spn, method="asymptotic")
        diffs[b] = np.nanmean(spp) - np.nanmean(spn)
        pVals[b] = p

    pVals_sorted = np.sort(pVals)
    ind_sorted = np.argsort(pVals)
    peakResp[n] = t[t >= 0][np.nanargmax(stimG[t >= 0])]
    posNeg[n] = np.nanmean(diffs) > 0
    for i in range(len(pVals_sorted)):
        alpha = 0.05 / (np.sum(binEnds < peakResp[n]) + 5 - 1 - i + 1)
        if pVals_sorted[i] < alpha:
            realBin = ind_sorted[i]
            posNeg[n] = diffs[realBin] > 0
            # maxDiffPoint[n] = binEnds[np.argmax(np.abs(diffs))]
            firstLatency = binEnds[realBin + 1]
            latencies[n] = firstLatency
            break
    # sigInd = np.where(pVals < (0.05) / nBins)[0]
    # if len(sigInd) > 0:
    #     firstLatency = binEnds[sigInd[0] + 1]
    #     posNeg[n] = diffs[sigInd[0]] > 0
    #     maxDiffPoint[n] = binEnds[np.argmax(np.abs(diffs))]
    #     latencies[n] = firstLatency
    peakResp[n] = t[(t >= 0) & (t <= 0.2)][
        np.nanargmax(stimG[(t >= 0) & (t <= 0.2)])
    ]
    diffsList.append(diffs)
    pvalsList.append(pVals)


tups = np.vstack((sessId, pId, nId)).T
tups = tups[r[0, :] == 1, :]


peakRespAll = np.zeros(tups.shape[0])


window = np.array([[-0.2, 1]])[0]
for n in range(tups.shape[0]):

    cellData = tups[n, :]
    p = AllData[cellData[0]]["probes"][cellData[1] + 1]

    imagingSide = p["ImagingSide"]

    scDepth = p["scDepth"][0]
    spikes = MakeSpikeInfoMatrix(p, [scDepth - 1000, scDepth])
    uniqueClust = np.unique(spikes[:, 1])
    spikesCell = spikes[spikes[:, 1] == uniqueClust[cellData[2]]]

    stimStarts = AllData[cellData[0]]["stimT"][:, 0]

    choice = AllData[cellData[0]]["choice"]
    feedback = AllData[cellData[0]]["feedbackType"]
    prevFb = np.append(0, feedback[:-1])

    if imagingSide == 1:
        contra = AllData[cellData[0]]["contrastLeft"][:, 0]
        ipsi = AllData[cellData[0]]["contrastRight"][:, 0]
        choice = -choice
    else:
        ipsi = AllData[cellData[0]]["contrastLeft"][:, 0]
        contra = AllData[cellData[0]]["contrastRight"][:, 0]

    relevantTrial = np.where(contra > 0)[0]
    choice = choice[relevantTrial]
    feedback = feedback[relevantTrial]
    stimStarts = stimStarts[relevantTrial]
    prevFb = prevFb[relevantTrial]

    stimA, t = GetRaster(spikesCell.copy(), stimStarts, window, dt=1 / 1000)
    stimA = GetPSTH(stimA[:, :, 0], w=5, halfGauss=True, flat=False)
    bl = np.nanmean(stimA[(t < 0), :], 0)

    stimG = np.nanmean(stimA[:, :], 1)

    peakRespAll[n] = t[(t >= 0) & (t <= 0.5)][
        np.nanargmax(stimG[(t >= 0) & (t <= 0.5)])
    ]

f, ax = plt.subplots(1)

ax.hist(
    peakRespAll, bins=np.arange(0, 0.500, 0.01), color="black", align="mid"
)
ax.hist(peakResp, bins=np.arange(0, 0.500, 0.01), color="purple", align="mid")

ax.set_xlabel("Peak Latency (s)")
ax.set_ylabel("Number")
ax.legend(["All", "Feedback Modulated"])
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
