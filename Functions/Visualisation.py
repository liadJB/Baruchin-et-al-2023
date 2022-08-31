# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 10:14:33 2022

@author: LABadmin
"""
import numpy as np 
from matplotlib import pyplot as plt
from matplotlib import rc
from matplotlib import cm
import random
import sklearn
import scipy as sp

from scipy import signal
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer   
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import ElasticNetCV
from sklearn.datasets import make_regression
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split

import os
import glob

from FileManagement import *
from FileManagement import MakeSpikeInfoMatrix
# from GlmModule import *
from DataProcessing import *
from StatTests import *

def PrintPopulationResponsivenessEphys(allData,iteration = 0,sortBy = [],specificSort = 0,addLines = False, depth = None,removeNonResp = True,w = 20):
    
    itRange = range(1,len(allData)+1)
    ts = []
    if (iteration!=0):
        itRange = iteration
    
    stimL = []
    stimR = []
    beep =  []
    moveL =  []
    moveR =  []
    feedbackP =  []
    feedbackN =  []
    lick =  []
    depths = []
    for i in itRange:
        
        movOnset = allData[i]['wheelMoveIntervals'][:,0]
        choice = allData[i]['wheelDisplacement'][:,0]
        choice[choice>0] = 1
        choice[choice<0] = -1
        
        # pupil_ds = AlignSignals(allData[i]['pupil'],allData[i]['pupilTimes'],allData[i]['calTimes'])
        
        stimStarts = allData[i]['stimT'][:,0] 
        stimEnds = allData[i]['stimT'][:,1]  
        
    
        ch = allData[i]['choice']
        beepStarts = allData[i]['goCueTimes']
        feedbackStarts = allData[i]['feedbackTimes']
        feedBackPeriod = allData[i]['posFeedbackPeriod']
        feedbackType = allData[i]['feedbackType'][:,0]
        
        window = np.array([-0.5, 0.5])
        
        # find proper onsets
        
        
        for pi in range(len(allData[i]['probes'])):
            p  = allData[i]['probes'][pi+1]
            imagingSide = p['ImagingSide']
            
            if (not 'times' in p.keys()) or (not 'scDepth' in p.keys()):
                continue;
            scDepth = p['scDepth'][0]
            
            spikes = MakeSpikeInfoMatrix(p,[scDepth-1000, scDepth])
            _,uniqueI = np.unique(spikes[:,1],return_index=True)
            depths.append(scDepth-spikes[uniqueI,-1])
            
            if (imagingSide==1):
                contra = allData[i]['contrastLeft'][:,0]
                ipsi = allData[i]['contrastRight'][:,0]
                choice = -choice
            else:
                ipsi = allData[i]['contrastLeft'][:,0]
                contra = allData[i]['contrastRight'][:,0]
            noStim = np.intersect1d(np.where(contra==0)[0],np.where(ipsi==0)[0])
            
            properMoveOnset = []
            properChoice = []

            for mi in range(len(movOnset)):
                m = movOnset[mi]
                c = choice[mi]
                # if (abs(allData[i]['wheelAmplitude'][:,0][mi])>(np.nanmean(abs(allData[i]['wheelAmplitude'][:,0][mi]))+1.5*np.nanstd(abs(allData[i]['wheelAmplitude'][:,0][mi])))):
                #     continue
                movT = np.intersect1d(np.where(m>stimStarts[noStim])[0],np.where(m<stimEnds[noStim])[0])
                if (len(movT)!=0):
                    properMoveOnset.append(m)
                    properChoice.append(c)
                # else:
                #     0
                
                # movT = np.intersect1d(np.where(m>stimStarts+0.2)[0],np.where(m<beepStarts)[0])
                # if (len(movT)!=0):
                #     properMoveOnset.append(m)
                #     properChoice.append(c)
                
            properMoveOnset = (np.array(properMoveOnset))
            properChoice = (np.array(properChoice))
            
            
            lickraw = allData[i]['lick_raw'][:,0]
            lickTimes = allData[i]['times']
            lcks = detectLicks(lickraw,distance = 200,height = 1)
            lcks = lickTimes[lcks]
            lcks = cleanLickBursts(lcks,0.3)
        
            # Take only licks that are not reward related
            removeLcks = []
            for lckI in range(len(lcks)):
                lck = lcks[lckI]
                timeFromReward = lck-feedbackStarts
                timeFromReward = timeFromReward[timeFromReward>0]
                if (len(timeFromReward)==0):
                    continue;
                shortestTimeFromReward = np.min(timeFromReward)
                # within reward time
                if (shortestTimeFromReward)<np.max(feedBackPeriod)+0.5: 
                    removeLcks.append(lckI)
            lcks = np.delete(lcks,removeLcks)
            
            
            # Aligned signals
            stimA,t = GetRaster(spikes.copy(), stimStarts, window ,dt=1/1000)
            beepA,t = GetRaster(spikes.copy(), beepStarts, window ,dt=1/1000)
            moveA,t = GetRaster(spikes.copy(), properMoveOnset, window ,dt=1/1000)  
            feedbackA,t = GetRaster(spikes.copy(), feedbackStarts, window ,dt=1/1000)
            lickA,t = GetRaster(spikes.copy(), lcks, window ,dt=1/1000)
            # ts.append(t)
            # rr2 = GetPSTH(ras[:,:,:],w=40)
            
            
            stimL.append(GetPSTH(stimA[:,((contra == np.max(contra)) & (ipsi==0)),:],w=w))
            stimR.append(GetPSTH(stimA[:,((ipsi == np.max(ipsi)) & (contra==0)),:],w=w))
            beep.append(GetPSTH(beepA[:,ch[:,0]==0,:],w=5))
            moveL.append(GetPSTH(moveA[:,properChoice == 1,:],w=w))
            moveR.append(GetPSTH(moveA[:,properChoice == -1,:],w=w))            
            feedbackP.append(GetPSTH(feedbackA[:,np.intersect1d(np.where(feedbackType == 1)[0],noStim),:],w=w))
            feedbackN.append(GetPSTH(feedbackA[:,np.intersect1d(np.where(feedbackType != 1)[0],noStim),:],w=w))
            lick.append(GetPSTH(lickA[:,:,:],w=w))
            # feedbackP.append(np.nanmean(feedbackA[:,feedbackType == 1,:],axis=1))
            # feedbackN.append(np.nanmean(feedbackA[:,feedbackType != 1,:],axis=1))
        
        
                
    rawResp = (stimL,stimR,beep,moveL,moveR,feedbackP,feedbackN,)
        
   
    
    stimL = sp.stats.zscore(np.hstack(stimL)*1000,axis=0,nan_policy='omit')
    stimR = sp.stats.zscore(np.hstack(stimR)*1000,axis=0,nan_policy='omit')
    beep = sp.stats.zscore(np.hstack(beep)*1000,axis=0,nan_policy='omit')
    moveL = sp.stats.zscore(np.hstack(moveL)*1000,axis=0,nan_policy='omit')
    moveR = sp.stats.zscore(np.hstack(moveR)*1000,axis=0,nan_policy='omit')
    feedbackP = sp.stats.zscore(np.hstack(feedbackP)*1000,axis=0,nan_policy='omit')
    feedbackN = sp.stats.zscore(np.hstack(feedbackN)*1000,axis=0,nan_policy='omit')
    lick = sp.stats.zscore(np.hstack(lick)*1000,axis=0,nan_policy='omit')
    depths = sp.hstack(depths)
    returnVal= (stimL,stimR,beep,moveL,moveR,feedbackP,feedbackN,lick,t,depths)
    # Remove NAN for visualisation
    Nans = np.where((np.nansum(stimL,axis=0)==0)| (np.nansum(stimR,axis=0)==0)|(np.nansum(beep,axis=0)==0)|(np.nansum(moveL,axis=0)==0)|(np.nansum(moveR,axis=0)==0)|(np.nansum(feedbackP,axis=0)==0)|(np.nansum(feedbackN,axis=0)==0))[0]
    
    if (removeNonResp):
        nonResp = np.where(np.sum(sortBy,0)==0)
        Nans = np.union1d(Nans,nonResp)
    if not(depth is None):
        wrongDepthInd = np.where((depths<=depth[0]) | (depths>=depth[1]))[0] # find neurons not in depth
        Nans = np.union1d(Nans,wrongDepthInd)
    
    stimL = np.delete(stimL,Nans,axis=1)
    stimR = np.delete(stimR,Nans,axis=1)
    beep = np.delete(beep,Nans,axis=1)
    moveL = np.delete(moveL,Nans,axis=1)
    moveR = np.delete(moveR,Nans,axis=1)
    feedbackP = np.delete(feedbackP,Nans,axis=1)
    feedbackN = np.delete(feedbackN,Nans,axis=1)
    lick = np.delete(lick,Nans,axis=1)
    depths = np.delete(depths,Nans,axis=0)    

    stimL = stimL - np.nanmean(stimL[(t<=0)&(t>-0.2),:],0)
    stimR = stimR - np.nanmean(stimR[(t<=0)&(t>-0.2),:],0)
    beep = beep - np.nanmean(beep[(t<=0)&(t>-0.2),:],0)
    moveL = moveL - np.nanmean(moveL[(t<=0)&(t>-0.2),:],0)
    moveR = moveR - np.nanmean(moveR[(t<=0)&(t>-0.2),:],0)
    feedbackP = feedbackP - np.nanmean(feedbackP[(t<=0)&(t>-0.2),:],0)
    feedbackN = feedbackN - np.nanmean(feedbackN[(t<=0)&(t>-0.2),:],0)
    lick = lick - np.nanmean(lick[(t<=0)&(t>-0.2),:],0)
    
    #Build sort column
    sortCol = np.zeros((8,stimL.shape[1]))
    sortCol[0,:] = np.nanmean(stimL[(t>0) & (t<=0.3),:],axis=0)
    sortCol[1,:]  = np.nanmean(stimR[(t>0) & (t<=0.3),:],axis=0)
    sortCol[2,:]  = np.nanmean(beep[(t>0) & (t<=0.3),:],axis=0)
    sortCol[3,:]  = np.nanmean(moveL[(t>0) & (t<=0.3),:],axis=0)
    sortCol[4,:]  = np.nanmean(moveR[(t>0) & (t<=0.3),:],axis=0)
    sortCol[5,:]  = np.nanmean(feedbackP[(t>0) & (t<=0.3),:],axis=0)
    sortCol[6,:]  = np.nanmean(feedbackN[(t>0) & (t<=0.3),:],axis=0)
    sortCol[7,:]  = np.nanmean(lick[(t>0) & (t<=0.3),:],axis=0)
    
    # find lims for visualisation
   
    c = [-4,4]
    depthInds = np.array([0,stimL.shape[1]])
    ind = np.flip(np.lexsort(sortCol))
       
    if (addLines):        
        lines = np.zeros(7)
        for i in range(7):
            sb = np.where(sortBy==i)[1]
            _,inInd1,inInd2 = np.intersect1d(ind,sb,return_indices = True)
            lines[i] = inInd1[-1]
            

    if (len(sortBy)!=0) :        
        sortBy = np.delete(sortBy,Nans,axis=1)
       
        # sorter[:,allZeros] = np.nan
        # sortPower[sortBy==0] = 0
        sorter = np.multiply(sortBy.astype(np.float64),sortCol)
        
        lastIndex = 0
        finalOrder = np.zeros(sortCol.shape[1])
        for si in range(sorter.shape[0]):
            # Take all what is not sorted as NAN
           
            sorter[si,sortBy[si,:]==0] = np.nan
            sortedArray = np.sort(sorter[si,:])
            FirstNan = np.where(np.isnan(sortedArray))[0][0]
            Ind = np.argsort(sorter[si,:])
            finalOrder[lastIndex:lastIndex+FirstNan] = np.flip(Ind[:FirstNan])
            lastIndex = lastIndex+FirstNan
            sorter[:,sortBy[si,:]==1] = np.nan
        
        finalOrder[lastIndex:] = np.where(np.sum(sortBy,0)==0)[0]
        ind = np.flip(finalOrder.astype(int))
     
    df = (t[1]-t[0])/2
    cmap = cm.seismic  
    cmap.set_bad('purple',1.)
        
    if (1==1):    
        fig0,ax = plt.subplots(1,2,figsize = (20,20))     
        fig0.subplots_adjust(wspace=0.1)               
        font = {'family' : 'Arial', 'size'   : 12}
        rc('font', **font)    
        fig0.suptitle('Stimulus Locked Response')
        if (specificSort):
            ind = np.flip(np.argsort(sortCol[0,:])).astype(int)
        im = ax[0].imshow(stimL[:,ind].T,extent = [t[0],t[-1],1,stimL.shape[1]],aspect='auto',cmap = 'seismic')
        im.set_clim(c)
        if (specificSort):
            ind = np.flip(np.argsort(sortCol[1,:])).astype(int)
        im = ax[1].imshow(stimR[:,ind].T,extent = [t[0],t[-1],1,stimL.shape[1]],aspect='auto',cmap = 'seismic')
        im.set_clim(c)
        ax[0].set_xlabel('Time')
        ax[1].set_xlabel('Time')
        ax[0].set_ylabel('Neurons')
        ax[1].set_ylabel('Neurons')
        ax[0].set_title('Contra')
        ax[1].set_title('Ipsi')
        ax[1].axes.yaxis.set_visible(False)
        ax[0].vlines(0,0,stimL.shape[1],colors = 'k',linestyles='dashed')
        ax[1].vlines(0,0,stimL.shape[1],colors = 'k',linestyles='dashed')
        ax[0].set_ylim(0,stimL.shape[1])
        ax[1].set_ylim(0,stimL.shape[1])
        ax[0].invert_yaxis()
        ax[1].invert_yaxis()    
        ax[0].set_xlim(-0.1,0.5)
        ax[1].set_xlim(-0.1,0.5)
        fig0.subplots_adjust(wspace=0.1)
        if (addLines): 
            ax[0].hlines(lines,t[0],t[-1],'k')
            ax[1].hlines(lines,t[0],t[-1],'k')
        
        
        
        fig1,ax = plt.subplots(1,2,figsize = (20,20))                    
        font = {'family' : 'Arial', 'size'   : 12}
        rc('font', **font)
        fig1.suptitle('Move Locked Response')
        if (specificSort):
            ind = np.flip(np.argsort(sortCol[2,:])).astype(int)
        im = ax[0].imshow(moveL[:,ind].T,extent = [t[0],t[-1],1,stimL.shape[1]],aspect='auto',cmap = 'seismic')
        im.set_clim(c)
        if (specificSort):
            ind = np.flip(np.argsort(sortCol[3,:])).astype(int)
        im = ax[1].imshow(moveR[:,ind].T,extent = [t[0],t[-1],1,stimL.shape[1]],aspect='auto',cmap = 'seismic')
        im.set_clim(c)
        ax[0].set_xlabel('Time')
        ax[1].set_xlabel('Time')
        ax[0].set_ylabel('Neurons')
        ax[1].set_ylabel('Neurons')
        ax[0].set_title('Move Contra')
        ax[1].set_title('Move Ipsi')
        ax[1].axes.yaxis.set_visible(False)
        ax[0].vlines(0,0,stimL.shape[1],colors = 'k',linestyles='dashed')
        ax[1].vlines(0,0,stimL.shape[1],colors = 'k',linestyles='dashed')
        fig1.subplots_adjust(wspace=0.1)
        ax[0].set_ylim(0,stimL.shape[1])
        ax[1].set_ylim(0,stimL.shape[1])
        ax[0].invert_yaxis()
        ax[1].invert_yaxis()
        ax[0].set_xlim(-0.1,0.5)
        ax[1].set_xlim(-0.1,0.5)
        if (addLines): 
            ax[0].hlines(lines,t[0],t[-1],'k')
            ax[1].hlines(lines,t[0],t[-1],'k')
        
        fig2,ax = plt.subplots(1,2,figsize = (20,20))                    
        font = {'family' : 'Arial', 'size'   : 12}
        rc('font', **font)   
        fig2.suptitle('Cue Locked Response')
        if (specificSort):
            ind = np.flip(np.argsort(sortCol[4,:])).astype(int)
        im = ax[0].imshow(beep[:,ind].T,extent = [t[0],t[-1],1,stimL.shape[1]],aspect='auto',cmap = 'seismic')
        im.set_clim(c)
        ax[0].set_xlabel('Time')
        ax[1].set_xlabel('Time')
        ax[0].set_ylabel('Neurons')
        ax[1].set_ylabel('Neurons')
        ax[0].set_title('Cue')
        ax[1].set_title('StimR')
        ax[1].set_visible(False)
        ax[0].vlines(0,0,stimL.shape[1],colors = 'k',linestyles='dashed')
        ax[1].vlines(0,0,stimL.shape[1],colors = 'k',linestyles='dashed')
        fig2.subplots_adjust(wspace=0.1)
        ax[0].set_ylim(0,stimL.shape[1])
        ax[1].set_ylim(0,stimL.shape[1])
        ax[0].invert_yaxis()
        ax[1].invert_yaxis()
        ax[0].set_xlim(-0.1,0.5)
        ax[1].set_xlim(-0.1,0.5)
        if (addLines): 
            ax[0].hlines(lines,t[0],t[-1],'k')
            ax[1].hlines(lines,t[0],t[-1],'k')
        
        fig3,ax = plt.subplots(1,2,figsize = (20,20))                    
        font = {'family' : 'Arial', 'size'   : 12}
        rc('font', **font) 
        fig3.suptitle('Feedback Locked Response')
        if (specificSort):
            ind = np.flip(np.argsort(sortCol[5,:])).astype(int)
        im = ax[0].imshow(feedbackP[:,ind].T,extent = [t[0],t[-1],1,stimL.shape[1]],aspect='auto',cmap = 'seismic')
        im.set_clim(c)
        if (specificSort):
            ind = np.flip(np.argsort(sortCol[6,:])).astype(int)
        im = ax[1].imshow(feedbackN[:,ind].T,extent = [t[0],t[-1],1,stimL.shape[1]],aspect='auto',cmap = 'seismic')
        im.set_clim(c)
        ax[0].set_xlabel('Time')
        ax[1].set_xlabel('Time')
        ax[0].set_ylabel('Neurons')
        ax[1].set_ylabel('Neurons')
        ax[0].set_title('Positive Feedback')
        ax[1].set_title('Negative Feedback')
        ax[1].axes.yaxis.set_visible(False)
        fig3.subplots_adjust(wspace=0.1)
        ax[0].vlines(0,0,stimL.shape[1],colors = 'k',linestyles='dashed')
        ax[1].vlines(0,0,stimL.shape[1],colors = 'k',linestyles='dashed')   
        ax[0].set_ylim(0,stimL.shape[1])
        ax[1].set_ylim(0,stimL.shape[1])
        ax[0].invert_yaxis()
        ax[1].invert_yaxis()
        ax[0].set_xlim(-0.1,0.5)
        ax[1].set_xlim(-0.1,0.5)
        if (addLines): 
            ax[0].hlines(lines,t[0],t[-1],'k')
            ax[1].hlines(lines,t[0],t[-1],'k')
        
        fig3.colorbar(im,ax=ax)
        
        fig4,ax = plt.subplots(1,2,figsize = (20,20))                    
        font = {'family' : 'Arial', 'size'   : 12}
        rc('font', **font)   
        fig4.suptitle('lick Locked Response')
        if (specificSort):
            ind = np.flip(np.argsort(sortCol[4,:])).astype(int)
        im = ax[0].imshow(lick[:,ind].T,extent = [t[0],t[-1],1,stimL.shape[1]],aspect='auto',cmap = 'seismic')
        im.set_clim(c)
        ax[0].set_xlabel('Time')
        ax[1].set_xlabel('Time')
        ax[0].set_ylabel('Neurons')
        ax[1].set_ylabel('Neurons')
        ax[0].set_title('Cue')
        ax[1].set_title('StimR')
        ax[1].set_visible(False)
        ax[0].vlines(0,0,stimL.shape[1],colors = 'k',linestyles='dashed')
        ax[1].vlines(0,0,stimL.shape[1],colors = 'k',linestyles='dashed')
        fig2.subplots_adjust(wspace=0.1)
        ax[0].set_ylim(0,stimL.shape[1])
        ax[1].set_ylim(0,stimL.shape[1])
        ax[0].invert_yaxis()
        ax[1].invert_yaxis()
        ax[0].set_xlim(-0.1,0.5)
        ax[1].set_xlim(-0.1,0.5)
        if (addLines): 
            ax[0].hlines(lines,t[0],t[-1],'k')
            ax[1].hlines(lines,t[0],t[-1],'k')
        
    else:
        fig0,ax = plt.subplots(3,2,figsize = (20,20))     
        fig0.subplots_adjust(wspace=0.1)               
        font = {'family' : 'Arial', 'size'   : 12}
        rc('font', **font)    
        fig0.suptitle('Stimulus Locked Response')
        if (specificSort):
            ind = np.flip(np.argsort(sortCol[0,:])).astype(int)
        for l in range(3):
            im = ax[l,0].imshow(stimL[:,ind].T,extent = [t[0],t[-1],1,stimL.shape[1]],aspect='auto',cmap = 'seismic')
            im.set_clim(c)
            if (specificSort):
                ind = np.flip(np.argsort(sortCol[1,:])).astype(int)
            im = ax[l,1].imshow(stimR[:,ind].T,extent = [t[0],t[-1],1,stimL.shape[1]],aspect='auto',cmap = 'seismic')
            im.set_clim(c)
            ax[l,0].set_xlabel('Time')
            ax[l,1].set_xlabel('Time')
            ax[l,0].set_ylabel('Neurons')
            ax[l,1].set_ylabel('Neurons')
            ax[l,0].set_title('Contra')
            ax[l,1].set_title('Ipsi')
            ax[l,1].axes.yaxis.set_visible(False)
            ax[l,0].vlines(0,0,stimL.shape[1],colors = 'k',linestyles='dashed')
            ax[l,1].vlines(0,0,stimL.shape[1],colors = 'k',linestyles='dashed')
            ax[l,0].set_ylim(0,stimL.shape[1])
            ax[l,1].set_ylim(0,stimL.shape[1])
            ax[l,0].invert_yaxis()
            ax[l,1].invert_yaxis()    
            fig0.subplots_adjust(wspace=0)
            if (addLines): 
                ax[l,0].hlines(lines,t[0],t[-1],'k')
                ax[l,1].hlines(lines,t[0],t[-1],'k')
        ax[0,0].set_ylim(depthInds[0,0],depthInds[1,0])
        ax[0,1].set_ylim(depthInds[0,0],depthInds[1,0])
        ax[1,0].set_ylim(depthInds[0,1],depthInds[1,1])
        ax[1,1].set_ylim(depthInds[0,1],depthInds[1,1])
        ax[2,0].set_ylim(depthInds[0,2],depthInds[1,2])
        ax[2,1].set_ylim(depthInds[0,2],depthInds[1,2])
                
        
        
        fig1,ax = plt.subplots(1,2,figsize = (20,20))                    
        font = {'family' : 'Arial', 'size'   : 12}
        rc('font', **font)
        fig1.suptitle('Move Locked Response')
        if (specificSort):
            ind = np.flip(np.argsort(sortCol[2,:])).astype(int)
        for l in range(3):
            im = ax[l,0].imshow(moveL[:,ind].T,extent = [t[0],t[-1],1,stimL.shape[1]],aspect='auto',cmap = 'seismic')
            im.set_clim(c)
            if (specificSort):
                ind = np.flip(np.argsort(sortCol[3,:])).astype(int)
            im = ax[l,1].imshow(moveR[:,ind].T,extent = [t[0],t[-1],1,stimL.shape[1]],aspect='auto',cmap = 'seismic')
            im.set_clim(c)
            ax[l,0].set_xlabel('Time')
            ax[l,1].set_xlabel('Time')
            ax[l,0].set_ylabel('Neurons')
            ax[l,1].set_ylabel('Neurons')
            ax[l,0].set_title('Move Contra')
            ax[l,1].set_title('Move Ipsi')
            ax[l,1].axes.yaxis.set_visible(False)
            ax[l,0].vlines(0,0,stimL.shape[1],colors = 'k',linestyles='dashed')
            ax[l,1].vlines(0,0,stimL.shape[1],colors = 'k',linestyles='dashed')
            fig1.subplots_adjust(wspace=0)
            ax[l,0].set_ylim(0,stimL.shape[1])
            ax[l,1].set_ylim(0,stimL.shape[1])
            ax[l,0].invert_yaxis()
            ax[l,1].invert_yaxis()
            if (addLines): 
                ax[l,0].hlines(lines,t[0],t[-1],'k')
                ax[l,1].hlines(lines,t[0],t[-1],'k')
        ax[0,0].set_ylim(depthInds[0,0],depthInds[1,0])
        ax[0,1].set_ylim(depthInds[0,0],depthInds[1,0])
        ax[1,0].set_ylim(depthInds[0,1],depthInds[1,1])
        ax[1,1].set_ylim(depthInds[0,1],depthInds[1,1])
        ax[2,0].set_ylim(depthInds[0,2],depthInds[1,2])
        ax[2,1].set_ylim(depthInds[0,2],depthInds[1,2])
        
        fig2,ax = plt.subplots(1,2,figsize = (20,20))                    
        font = {'family' : 'Arial', 'size'   : 12}
        rc('font', **font)   
        fig2.suptitle('Cue Locked Response')
        if (specificSort):
            ind = np.flip(np.argsort(sortCol[4,:])).astype(int)
        for l in range(3):
            im = ax[l,0].imshow(beep[:,ind].T,extent = [t[0],t[-1],1,stimL.shape[1]],aspect='auto',cmap = 'seismic')
            im.set_clim(c)
            ax[l,0].set_xlabel('Time')
            ax[l,1].set_xlabel('Time')
            ax[l,0].set_ylabel('Neurons')
            ax[l,1].set_ylabel('Neurons')
            ax[l,0].set_title('Cue')
            ax[l,1].set_title('StimR')
            ax[l,1].set_visible(False)
            ax[l,0].vlines(0,0,stimL.shape[1],colors = 'k',linestyles='dashed')
            ax[l,1].vlines(0,0,stimL.shape[1],colors = 'k',linestyles='dashed')
            fig2.subplots_adjust(wspace=0)
            ax[l,0].set_ylim(0,stimL.shape[1])
            ax[l,1].set_ylim(0,stimL.shape[1])
            ax[l,0].invert_yaxis()
            ax[l,1].invert_yaxis()
            if (addLines): 
                ax[l,0].hlines(lines,t[0],t[-1],'k')
                ax[l,1].hlines(lines,t[0],t[-1],'k')
        ax[0,0].set_ylim(depthInds[0,0],depthInds[1,0])
        ax[0,1].set_ylim(depthInds[0,0],depthInds[1,0])
        ax[1,0].set_ylim(depthInds[0,1],depthInds[1,1])
        ax[1,1].set_ylim(depthInds[0,1],depthInds[1,1])
        ax[2,0].set_ylim(depthInds[0,2],depthInds[1,2])
        ax[2,1].set_ylim(depthInds[0,2],depthInds[1,2])
        
        fig3,ax = plt.subplots(1,2,figsize = (20,20))                    
        font = {'family' : 'Arial', 'size'   : 12}
        rc('font', **font) 
        fig3.suptitle('Feedback Locked Response')
        if (specificSort):
            ind = np.flip(np.argsort(sortCol[5,:])).astype(int)
        for l in range(3):
            im = ax[l,0].imshow(feedbackP[:,ind].T,extent = [t[0],t[-1],1,stimL.shape[1]],aspect='auto',cmap = 'seismic')
            im.set_clim(c)
            if (specificSort):
                ind = np.flip(np.argsort(sortCol[6,:])).astype(int)
            im = ax[l,1].imshow(feedbackN[:,ind].T,extent = [t[0],t[-1],1,stimL.shape[1]],aspect='auto',cmap = 'seismic')
            im.set_clim(c)
            ax[l,0].set_xlabel('Time')
            ax[l,1].set_xlabel('Time')
            ax[l,0].set_ylabel('Neurons')
            ax[l,1].set_ylabel('Neurons')
            ax[l,0].set_title('Positive Feedback')
            ax[l,1].set_title('Negative Feedback')
            ax[l,1].axes.yaxis.set_visible(False)
            fig3.subplots_adjust(wspace=0)
            ax[l,0].vlines(0,0,stimL.shape[1],colors = 'k',linestyles='dashed')
            ax[l,1].vlines(0,0,stimL.shape[1],colors = 'k',linestyles='dashed')   
            ax[l,0].set_ylim(0,stimL.shape[1])
            ax[l,1].set_ylim(0,stimL.shape[1])
            ax[l,0].invert_yaxis()
            ax[l,1].invert_yaxis()
            if (addLines): 
                ax[l,0].hlines(lines,t[0],t[-1],'k')
                ax[l,1].hlines(lines,t[0],t[-1],'k')
        ax[0,0].set_ylim(depthInds[0,0],depthInds[1,0])
        ax[0,1].set_ylim(depthInds[0,0],depthInds[1,0])
        ax[1,0].set_ylim(depthInds[0,1],depthInds[1,1])
        ax[1,1].set_ylim(depthInds[0,1],depthInds[1,1])
        ax[2,0].set_ylim(depthInds[0,2],depthInds[1,2])
        ax[2,1].set_ylim(depthInds[0,2],depthInds[1,2])        
        fig3.colorbar(im,ax=ax)
        
        fig4,ax = plt.subplots(1,2,figsize = (20,20))                    
        font = {'family' : 'Arial', 'size'   : 12}
        rc('font', **font) 
        fig4.suptitle('Lick Locked Response')
        if (specificSort):
            ind = np.flip(np.argsort(sortCol[5,:])).astype(int)
        for l in range(3):
            im = ax[l,0].imshow(feedbackP[:,ind].T,extent = [t[0],t[-1],1,stimL.shape[1]],aspect='auto',cmap = 'seismic')
            im.set_clim(c)
            if (specificSort):
                ind = np.flip(np.argsort(sortCol[6,:])).astype(int)
            im = ax[l,1].imshow(feedbackN[:,ind].T,extent = [t[0],t[-1],1,stimL.shape[1]],aspect='auto',cmap = 'seismic')
            im.set_clim(c)
            ax[l,0].set_xlabel('Time')
            ax[l,1].set_xlabel('Time')
            ax[l,0].set_ylabel('Neurons')
            ax[l,1].set_ylabel('Neurons')
            ax[l,0].set_title('Positive Feedback')
            ax[l,1].set_title('Negative Feedback')
            ax[l,1].axes.yaxis.set_visible(False)
            fig3.subplots_adjust(wspace=0)
            ax[l,0].vlines(0,0,stimL.shape[1],colors = 'k',linestyles='dashed')
            ax[l,1].vlines(0,0,stimL.shape[1],colors = 'k',linestyles='dashed')   
            ax[l,0].set_ylim(0,stimL.shape[1])
            ax[l,1].set_ylim(0,stimL.shape[1])
            ax[l,0].invert_yaxis()
            ax[l,1].invert_yaxis()
            if (addLines): 
                ax[l,0].hlines(lines,t[0],t[-1],'k')
                ax[l,1].hlines(lines,t[0],t[-1],'k')
        ax[0,0].set_ylim(depthInds[0,0],depthInds[1,0])
        ax[0,1].set_ylim(depthInds[0,0],depthInds[1,0])
        ax[1,0].set_ylim(depthInds[0,1],depthInds[1,1])
        ax[1,1].set_ylim(depthInds[0,1],depthInds[1,1])
        ax[2,0].set_ylim(depthInds[0,2],depthInds[1,2])
        ax[2,1].set_ylim(depthInds[0,2],depthInds[1,2])        
        fig4.colorbar(im,ax=ax)
        
        
    return returnVal

def getResponsivnessStats(resp):
    keys = ['Contra','Ipsi','Cue','Move Contra','Move Ipsi','Positive Feedback','Negative Feedback','licking']
    
    # Get shared responses
    sharedResp = np.zeros((8,8))
    respCount= np.zeros(8)
    for i in range(8):             
        for j in range(8):
            sharedResp[i,j] = np.sum(resp[i,:] & resp[j,:])
            if (i>j):
                sharedResp[j,i] =np.nan
    font = {'family' : 'Arial', 'size'   : 12}
    rc('font', **font)
    
    figs = []
    
    fig,ax = plt.subplots(1)
    maxResp = np.nanmax(sharedResp)
    im = ax.imshow(sharedResp, cmap = 'bone_r')
    im.set_clim(0,maxResp)
    ax.set_xticks(np.arange(len(keys)))
    ax.set_yticks(np.arange(len(keys)))    
    ax.set_xticklabels(keys,rotation=20)
    ax.set_yticklabels(keys )
    fig.colorbar(im,ax = ax)
    ax.set_frame_on(False)
    fig.suptitle('Absolute cooccurence')
    figs.append(fig)
    
    fig,ax = plt.subplots(1)
    diag_ = np.array(np.diag(sharedResp))
    respCount = np.array(np.diag(sharedResp))
    diag_[diag_==0]=1
    im = ax.imshow(sharedResp/diag_, cmap = 'bone_r')
    ax.set_xticks(np.arange(len(keys)))
    ax.set_yticks(np.arange(len(keys)))    
    ax.set_xticklabels(keys,rotation=20)
    ax.set_yticklabels(keys )
    ax.set_frame_on(False)
    fig.colorbar(im,ax = ax)
    fig.suptitle('Normalised cooccurence')
    figs.append(fig)
    
    fig,ax = plt.subplots(1)
    
    ax.bar(range(8),respCount)
    ax.set_xticks(np.arange(len(keys)))
    ax.set_xticklabels(keys,rotation=20 )
    ax.set_frame_on(False)
    fig.suptitle('Number of Responsive Neurons /'+str(resp.shape[1]))
    figs.append(fig)
    
    uniqueResp = resp[:,np.sum(resp,0)==1]
    uniqueRespCount = np.sum(uniqueResp,1)
    fig,ax = plt.subplots(1)    
    ax.bar(range(8),uniqueRespCount)
    ax.set_xticks(np.arange(len(keys)))
    ax.set_xticklabels(keys,rotation=20 )
    ax.set_frame_on(False)
    fig.suptitle('Number of Uniquely Responsive Neurons /'+str(uniqueResp.shape[1]))
    ax.set_ylim(0,max(respCount))
    figs.append(fig)
    
    nonMoveResp = resp[:,np.sum(resp[3:5,:],0)==0]
    nonMoveRespCount = np.sum(nonMoveResp,1)
    fig,ax = plt.subplots(1)    
    ax.bar(range(8),nonMoveRespCount)
    ax.set_xticks(np.arange(len(keys)))
    ax.set_xticklabels(keys,rotation=20 )
    ax.set_frame_on(False)
    fig.suptitle('Number of Non-Moving Neurons /'+str(nonMoveResp.shape[1]))
    ax.set_ylim(0,max(respCount))
    
    figs.append(fig)
    numResponses = np.sum(resp,0)
    fig,ax = plt.subplots(1)  
    fig.suptitle('Distribution of response to events')
    ax.hist(numResponses,[-1,0,1,2,3,4,5,6,7,8,9],color = 'black')
    ax.set_frame_on(False)
    figs.append(fig)
    return respCount,sharedResp,figs

def PrintResponseByAllGroups(allData,iteration=0,nn=[], specificCases = 'None',saveDir = 'D:\\Figures\\AllNeuronResponse\\'):
    font = {'family' : 'Arial', 'size'   : 14}
        
    rc('font', **font)
    
    if not os.path.isdir(saveDir):
      os.makedirs(saveDir)
      
    itRange = range(1,len(allData)+1)
    ts = []
    if (iteration!=0):
        itRange = iteration

    if (specificCases=='licking'):            
            saveDir = saveDir[:-1] + '_NoStimLick\\'
            if not os.path.isdir(saveDir):
                os.makedirs(saveDir)
                
    if (specificCases=='noGo'):      
        saveDir = saveDir[:-1] + '_NoDisengagement\\'
        if not os.path.isdir(saveDir):
            os.makedirs(saveDir)
            
    for i in itRange:
    
        stimStarts = allData[i]['stimT'][:,0]
        imagingSide= allData[i]['ImagingSide']  
        choice = allData[i]['choice']
        if (imagingSide==1):
            contra = allData[i]['contrastLeft'][:,0]
            ipsi = allData[i]['contrastRight'][:,0]
            choice*=-1
        else:
            ipsi = allData[i]['contrastLeft'][:,0]
            contra = allData[i]['contrastRight'][:,0]    
        conDiffs = contra-ipsi    
        
        uniqueConContra = np.unique(contra)
        uniqueConIpsi = np.unique(ipsi)
        
        choice = allData[i]['choice']
        feedback = allData[i]['feedbackType']
        
        delays = allData[i]['planeDelays']
        planes = allData[i]['planes'] 
        times = allData[i]['calTimes'] 
        sig = allData[i]['calTrace']
        
        negTrials = np.where(feedback<=0)[0]
        posTrials = np.where(feedback==1)[0]
        changeFeedback = np.where(np.diff(feedback,axis=0)!=0)[0]+1 #the very first possible value is the second trial
        
        window = np.array([[-1, 1]])  
        
        # Pupil
        pupil_ds = AlignSignals(allData[i]['pupil'],allData[i]['pupilTimes'],allData[i]['calTimes'])
        middle = np.nanmedian(pupil_ds,0)
        pu,t = AlignStim(pupil_ds, times, stimStarts, window)        
        bigTrials = pu>middle
        bigTrialsSum = np.sum(bigTrials[:,:,0],0)
        meanPupil = np.nanmean(pu[:,:,0],0)
        largePupil = (bigTrialsSum/len(t))>0.5
        smallPupil = (bigTrialsSum/len(t))<0.5   
    
       
        uniqueConDif = np.unique(conDiffs)   
       
        
        ca_raw,t = GetCalciumAligned(sig, times, stimStarts, window,planes,delays)
        ca_raw = ca_raw/np.max(np.max(ca_raw,1),0)
        ca = ca_raw - np.tile(np.mean(ca_raw[t<=0,:,:],axis=0),(ca_raw.shape[0],1,1))
        
        
        
        
        influences = ['none','pupil','go','correctness','feedback']       
        
        n_number = range(ca.shape[2])
        if len(nn)>0:
           n_number = nn
        for n in n_number:
            fig,ax = plt.subplots(5,len(uniqueConDif),figsize = (20,20))
            
            
            for inf in range(len(influences)):
                influence = influences[inf]
                
                g1Name = 'Positive Feedback'
                g2Name = 'Negative Feedback'
                
                if (influence == 'none'):
                    posBefore = range(ca.shape[1])
                    negBefore = range(ca.shape[1])
                    
                
                if (influence == 'feedback'):
                    posBefore = posTrials + 1
                    negBefore = negTrials + 1
                    posBefore = posBefore[[posBefore<len(feedback)]]
                    negBefore = negBefore[[negBefore<len(feedback)]]       
                    
                if (influence == 'correctness'): 
                    g1Name = 'correct'
                    g2Name = 'incorrect'
                   
                    move = np.where(choice!=0)[0]
                    posBefore = posTrials
                    negBefore = negTrials
                    
                    posBefore = np.intersect1d(move,posBefore)
                    negBefore = np.intersect1d(move,negBefore)
                    
                if (influence == 'go'): 
                    g1Name = 'Go'
                    g2Name = 'No Go'
                    # fazedOutSum
                    
                    posBefore = np.where(choice!=0)[0]
                    negBefore = np.where(choice==0)[0]     
                    
                if (influence == 'pupil'):   
                    g1Name = 'Large Pupil'
                    g2Name = 'Small pupil'
                    posBefore = np.where(largePupil)[0]
                    negBefore = np.where(smallPupil)[0]                
                
                 #####REMOVE LICKING TRIALS
                if (specificCases=='licking'):
                    lickraw = allData[i]['lick_raw'][:,0]
                    lickTimes = allData[i]['wheelTimes']
                    lickRate = getLickRate(lickraw,lickTimes)
                    lckStim,tt = AlignStim(lickRate, lickTimes, stimStarts, window) 
                    
                    testTime = np.where((tt>0.01) & (tt<=0.5))[0]
                    g1 = lckStim[:,:,0]
                    g1 = np.mean(g1[testTime],0)
                    noLick = np.where(g1==0)[0]
                    
                    posBefore = np.intersect1d(posBefore,noLick)
                    negBefore  = np.intersect1d(negBefore,noLick)
                     
                   
                    
                if (specificCases=='noGo'):
                    disengagedInds = []
                    noGoInds = np.where(choice==0)[0]
                    
                    for ng in noGoInds:
                        if (ng>2) & (ng-1) & (ng-2):
                            disengagedInds.append(ng)
                
            
          
                    posBefore = np.setdiff1d(posBefore,disengagedInds)
                    negBefore  = np.setdiff1d(negBefore,disengagedInds)
                
                _printSingleRow(ax[inf,:],ca_raw,posBefore,negBefore,conDiffs,n,t,1)
                if influence == 'none':
                    ax[inf,-1].legend(['All Trials'],loc=(1.1,0.5), fontsize='small')
                else:
                    ax[inf,-1].legend([g1Name,g2Name],loc=(1.1,0.5), fontsize='small')
                
            for ci in range(len(uniqueConDif)):     
                c = uniqueConDif[ci]
                ax[0,ci].set_title('Contrast: '+str(c))
            ax[0,0].set_ylabel('No Split')
            
            ax[1,0].set_ylabel('Pupil Size')
            ax[2,0].set_ylabel('Go')
            ax[3,0].set_ylabel('Correct')
            ax[4,0].set_ylabel('Prev. Feedback')
            fig.text(0.5, 0.04, 'Time Post-Stimulus (s)', ha='center')
            fig.text(0.04, 0.5, 'Normalised dF/F', va='center', rotation='vertical')
            plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.5, hspace=0.2)
            
            plt.savefig(saveDir+'Session'+str(i)+'_Neuron'+str(n)+'.png')
            plt.savefig(saveDir+'Session'+str(i)+'_Neuron'+str(n)+'.pdf', format='pdf')
            plt.close()
            
def _printSingleRow(ax,ca,posBefore,negBefore,conDiffs,n,t,cmax = 1):
        uniqueConDif = np.unique(conDiffs)     
                    
        ci = 0  
        g1ConValues = []
        g2ConValues = []
        g1IpsiValues = []
        g2IpsiValues = []
       
        for ci in range(len(uniqueConDif)):
            c = uniqueConDif[ci]
            Ind = np.intersect1d(np.where((conDiffs==c)),posBefore)
            y1 = np.nanmean(ca[:,Ind,n],1) #ca_raw
            err1 = np.nanstd(ca[:,Ind,n],1)/np.sqrt(len(Ind))            
            Ind = np.intersect1d(np.where((conDiffs==c)),negBefore)
            y2 = np.nanmean(ca[:,Ind,n],1)
            err2 = np.nanstd(ca[:,Ind,n],1)/np.sqrt(len(Ind))          
            
            if (np.sum(np.array(posBefore)==np.array(negBefore))==len(posBefore)):
                ax[ci].plot(t,y1,'k')
                ax[ci].fill_between(t,y1-err1,y1+err1,alpha=0.5,facecolor='k')
            else:                
                ax[ci].plot(t,y1,'b')
                ax[ci].fill_between(t,y1-err1,y1+err1,alpha=0.5,facecolor='b')
                ax[ci].plot(t,y2,'r')
                ax[ci].fill_between(t,y2-err2,y2+err2,alpha=0.5,facecolor='r')
            if np.isnan(cmax):
                cmax = 1
            ax[ci].set_ylim([0,cmax])    
            ax[ci].set_frame_on(False)
            
def PrintPopulationResponsiveness(allData,iteration = 0,sortBy = [],specificSort = False,addLines = False, BLsub = True, removeNonResp = True, focusOne=None):
    
    itRange = range(1,len(allData)+1)
    ts = []
    if (iteration!=0):
        itRange = iteration
    
    stimL = []
    stimR = []
    beep =  []
    moveL =  []
    moveR =  []
    feedbackP =  []
    feedbackN =  []
    lick =  []
    stimLsd = []
    stimRsd = []
    beepsd =  []
    moveLsd =  []
    moveRsd =  []
    feedbackPsd =  []
    feedbackNsd =  []
    licksd =  []
    for i in itRange:
        wheel_ds = AlignSignals(allData[i]['wheelVelocity'][::100,:],allData[i]['wheelTimes'][::100,:],allData[i]['calTimes'])        
        movOnset,choice = CreateMovementOnsetSignal(wheel_ds, allData[i]['calTimes'])
        choice = choice[:,0]
        pupil_ds = AlignSignals(allData[i]['pupil'],allData[i]['pupilTimes'],allData[i]['calTimes'])
        sig = allData[i]['calTrace']     
        #normalise sig      
        sig = sp.stats.zscore(sig,axis=0,nan_policy='omit')
        delays = allData[i]['planeDelays']
        planes = allData[i]['planes']               
        times = allData[i]['calTimes']       
        stimStarts = allData[i]['stimT'][:,0] 
        stimEnds = allData[i]['stimT'][:,1]   
        imagingSide= allData[i]['ImagingSide']   
        if (imagingSide==1):
            contra = allData[i]['contrastLeft'][:,0]
            ipsi = allData[i]['contrastRight'][:,0]
            choice = -choice
        else:
            ipsi = allData[i]['contrastLeft'][:,0]
            contra = allData[i]['contrastRight'][:,0]
    
        ch = allData[i]['choice'][:,0]
        beepStarts = allData[i]['goCueTimes']
        feedbackStarts = allData[i]['feedbackTimes']
        feedBackPeriod = allData[i]['posFeedbackPeriod']
        feedbackType = allData[i]['feedbackType'][:,0]
        
        window = np.array([[-0.5, 1]])
        
        # find proper onsets
        properMoveOnset = []
        properChoice = []
        noStim = np.intersect1d(np.where(contra==0)[0],np.where(ipsi==0)[0])
        for mi in range(len(movOnset)):
            m = movOnset[mi]
            c = choice[mi]
           
            movT = np.intersect1d(np.where(m>stimStarts+0.2)[0],np.where(m<beepStarts)[0])
            if (len(movT)!=0):
                properMoveOnset.append(m)
                properChoice.append(c)
            
        properMoveOnset = (np.array(properMoveOnset))
        properChoice = (np.array(properChoice))
        
        lickraw = allData[i]['lick_raw'][:,0]
        lickTimes = allData[i]['wheelTimes']
        lcks = detectLicks(lickraw)
        lcks = lickTimes[lcks]        
        lcks = cleanLickBursts(lcks,0.3)
                
            
        
        # Take only licks that are not reward related
        removeLcks = []
        for lckI in range(len(lcks)):
            lck = lcks[lckI]
            timeFromReward = lck-feedbackStarts
            timeFromReward = timeFromReward[timeFromReward>0]
            if (len(timeFromReward)==0):
                continue;
            shortestTimeFromReward = np.min(timeFromReward)
            # within reward time
            if (shortestTimeFromReward)<np.max(feedBackPeriod)+0.5: 
                removeLcks.append(lckI)
        lcks = np.delete(lcks,removeLcks)
        
        # Aligned signals
        stimA,t = GetCalciumAligned(sig, times, stimStarts, window,planes,delays)
        beepA,t = GetCalciumAligned(sig, times, beepStarts, window,planes,delays)
        moveA,t = GetCalciumAligned(sig, times, properMoveOnset, window,planes,delays)
        feedbackA,t = GetCalciumAligned(sig, times, feedbackStarts, window,planes,delays)
        lickA,t = GetCalciumAligned(sig, times, lcks, window,planes,delays)
        ts.append(t)
        
        if (BLsub):
            stimL.append(np.nanmean(stimA[:,((contra==np.max(contra))),:]-np.nanmean(stimA[(t<0)&(t>=-0.2),:,:][:,((contra==np.max(contra))),:],0),axis=1))
            stimR.append(np.nanmean(stimA[:,((ipsi>0) & (ch==0)),:]-np.nanmean(stimA[(t<0)&(t>=-0.2),:,:][:,((ipsi>0) & (ch==0)),:],0),axis=1))
            beep.append(np.nanmean(beepA[:,ch==0,:]-np.nanmean(beepA[(t<0)&(t>=-0.2),:,:][:,ch==0,:],0),axis=1))
            # moveL.append(np.nanmean(moveA[:,properChoice == 1,:]-np.nanmean(moveA[t<0,:,:][:,properChoice == 1,:],0),axis=1))
            # moveR.append(np.nanmean(moveA[:,properChoice == -1,:]-np.nanmean(moveA[t<0,:,:][:,properChoice == -1,:],0),axis=1))        
            moveL.append(np.nanmean(moveA[:,properChoice == 1,:]-np.nanmean(moveA[(t<0)&(t>=-0.2),:,:][:,properChoice == 1,:],0),axis=1))
            moveR.append(np.nanmean(moveA[:,properChoice == -1,:]-np.nanmean(moveA[t<0,:,:][:,properChoice == -1,:],0),axis=1))
            fb = feedbackA[:,:,:]-np.nanmean(feedbackA[(t<0)&(t>=-0.2),:,:],0)
            feedbackP.append(np.nanmean(fb[:,np.intersect1d(np.where(feedbackType == 1)[0],noStim),:],1))
            feedbackN.append(np.nanmean(fb[:,np.intersect1d(np.where(feedbackType != 1)[0],noStim),:],1))
            # feedbackP.append(np.nanmean(feedbackA[:,np.intersect1d(np.where(feedbackType == 1)[0],noStim),:]-np.nanmean(feedbackA[t<0,:,:][:,np.intersect1d(np.where(feedbackType == 1)[0],noStim),:],0),axis=1))
            # feedbackN.append(np.nanmean(feedbackA[:,np.intersect1d(np.where(feedbackType != 1)[0],noStim),:]-np.nanmean(feedbackA[t<0,:,:][:,np.intersect1d(np.where(feedbackType != 1)[0],noStim),:],0),axis=1))
            
            # feedbackP.append(np.nanmean(feedbackA[:,feedbackType == 1,:],axis=1))
            # feedbackN.append(np.nanmean(feedbackA[:,feedbackType != 1,:],axis=1))
            lick.append(np.nanmean(lickA[:,:,:]-np.nanmean(lickA[(t<0)&(t>=-0.2),:,:][:,:,:],0),axis=1))
        else:
            stimL.append(np.nanmean(stimA[:,((contra==np.max(contra))),:],axis=1))
            stimR.append(np.nanmean(stimA[:,((ipsi>0) & (ch==0)),:],axis=1))
            beep.append(np.nanmean(beepA[:,ch==0,:],axis=1))
            moveL.append(np.nanmean(moveA[:,properChoice == 1,:],axis=1))
            moveR.append(np.nanmean(moveA[:,properChoice == -1,:],axis=1))        
            feedbackP.append(np.nanmean(feedbackA[:,np.intersect1d(np.where(feedbackType == 1)[0],noStim),:],axis=1))
            feedbackN.append(np.nanmean(feedbackA[:,np.intersect1d(np.where(feedbackType != 1)[0],noStim),:],axis=1))
            # feedbackP.append(np.nanmean(feedbackA[:,feedbackType == 1,:],axis=1))
            # feedbackN.append(np.nanmean(feedbackA[:,feedbackType != 1,:],axis=1))
            lick.append(np.nanmean(lickA[:,:,:],axis=1))
        
        stimLsd.append(np.nanstd(stimA[:,((contra>0) & (ch==0)),:],axis=1)/np.sqrt(np.sum(((contra>0) & (ch==0)))))
        stimRsd.append(np.nanstd(stimA[:,((ipsi>0) & (ch==0)),:],axis=1)/np.sqrt(np.sum(((ipsi>0) & (ch==0)))))
        beepsd.append(np.nanstd(beepA[:,ch==0,:],axis=1)/np.sqrt(np.sum(ch==0)))
        moveLsd.append(np.nanstd(moveA[:,properChoice == 1,:],axis=1)/np.sqrt(np.sum(properChoice == 1)))
        moveRsd.append(np.nanstd(moveA[:,properChoice == -1,:],axis=1)/np.sqrt(np.sum(properChoice == -1)))        
        feedbackPsd.append(np.nanstd(feedbackA[:,np.intersect1d(np.where(feedbackType == 1)[0],noStim),:],axis=1)/np.sqrt(len(np.intersect1d(np.where(feedbackType == 1)[0],np.where(contra==0)[0]))))
        feedbackNsd.append(np.nanstd(feedbackA[:,np.intersect1d(np.where(feedbackType != 0)[0],noStim),:],axis=1)/np.sqrt(len(np.intersect1d(np.where(feedbackType != 1)[0],np.where(contra==0)[0]))))
        licksd.append(np.nanstd(beepA[:,:,:],axis=1)/np.sqrt(lickA.shape[1]))
    
    # Need to downsample some signals
    minSamples = 1000
    slowestT = []
    for tt in ts:
        dur = len(tt)
        if (dur<minSamples):
            minSamples = dur
            slowestT = tt # this will be what others will downsample to
            
    rawResp = (stimL,stimR,beep,moveL,moveR,feedbackP,feedbackN,ts)
        
    t = slowestT
    for i in range(len(stimL)):
        if len(ts[i]) == minSamples:
            continue;
        stimL[i] = AlignSignals(stimL[i],ts[i],slowestT,False)
        stimR[i] = AlignSignals(stimR[i],ts[i],slowestT,False)
        beep[i] = AlignSignals(beep[i],ts[i],slowestT,False)
        moveL[i] = AlignSignals(moveL[i],ts[i],slowestT,False)
        moveR[i] = AlignSignals(moveR[i],ts[i],slowestT,False)
        feedbackP[i] = AlignSignals(feedbackP[i],ts[i],slowestT,False)
        feedbackN[i] = AlignSignals(feedbackN[i],ts[i],slowestT,False)
        lick[i] = AlignSignals(lick[i],ts[i],slowestT,False)
        
        stimLsd[i] = AlignSignals(stimLsd[i],ts[i],slowestT)
        stimRsd[i] = AlignSignals(stimRsd[i],ts[i],slowestT)
        beepsd[i] = AlignSignals(beepsd[i],ts[i],slowestT)
        moveLsd[i] = AlignSignals(moveLsd[i],ts[i],slowestT)
        moveRsd[i] = AlignSignals(moveRsd[i],ts[i],slowestT)
        feedbackPsd[i] = AlignSignals(feedbackPsd[i],ts[i],slowestT)
        feedbackNsd[i] = AlignSignals(feedbackNsd[i],ts[i],slowestT)
        licksd[i] = AlignSignals(licksd[i],ts[i],slowestT)
    
    
    stimL = np.hstack(stimL)
    stimR = np.hstack(stimR)
    beep = np.hstack(beep)
    moveL = np.hstack(moveL)
    moveR = np.hstack(moveR)
    feedbackP = np.hstack(feedbackP)
    feedbackN = np.hstack(feedbackN)    
    lick = np.hstack(lick)
    
    stimLsd = np.hstack(stimLsd)
    stimRsd = np.hstack(stimRsd)
    beepsd = np.hstack(beepsd)
    moveLsd = np.hstack(moveLsd)
    moveRsd = np.hstack(moveRsd)
    feedbackPsd = np.hstack(feedbackPsd)
    feedbackNsd = np.hstack(feedbackNsd)
    licksd = np.hstack(licksd)
    
    
    # BL subtract
    BL = (np.nanmean(stimL[t<0,:],axis=0) + np.nanmean(stimR[t<0,:],axis=0))/2
    # stimL = stimL - np.nanmean(stimL[t<0,:],axis=0)
    # stimR = stimR - np.nanmean(stimR[t<0,:],axis=0)
    # beep = beep - np.nanmean(beep[t<0,:],axis=0)
    # moveL = moveL - np.nanmean(moveL[t<0,:],axis=0)
    # moveR = moveR - np.nanmean(moveR[t<0,:],axis=0)
    # feedbackP = feedbackP - np.nanmean(feedbackP[t<0,:],axis=0)
    # feedbackN = feedbackN - np.nanmean(feedbackN[t<0,:],axis=0)
    stimL = stimL - BL
    stimR = stimR - BL
    beep = beep - BL
    moveL = moveL - BL
    moveR = moveR - BL
    feedbackP = feedbackP - BL
    feedbackN = feedbackN - BL
    lick = lick - BL
    
    returnVal= (stimL,stimR,beep,moveL,moveR,feedbackP,feedbackN,lick,t)
    # Remove NAN for visualisation
    Nans = np.where((np.isnan(np.sum(stimL,axis=0)))| (np.isnan(np.sum(stimR,axis=0)))|(np.isnan(np.sum(beep,axis=0)))|(np.isnan(np.sum(moveL,axis=0)))|(np.isnan(np.sum(moveR,axis=0)))|(np.isnan(np.sum(feedbackP,axis=0)))|(np.isnan(np.sum(feedbackN,axis=0))))[0]
    
    if (removeNonResp):
        nonResp = np.where(np.sum(sortBy,0)==0)
        Nans = np.union1d(Nans,nonResp)
    
    stimL = np.delete(stimL,Nans,axis=1)
    stimR = np.delete(stimR,Nans,axis=1)
    beep = np.delete(beep,Nans,axis=1)
    moveL = np.delete(moveL,Nans,axis=1)
    moveR = np.delete(moveR,Nans,axis=1)
    feedbackP = np.delete(feedbackP,Nans,axis=1)
    feedbackN = np.delete(feedbackN,Nans,axis=1)
    lick = np.delete(lick,Nans,axis=1)
    
    
    #Build sort column
    sortCol = np.zeros((8,stimL.shape[1]))
    sortCol[0,:] = np.nanmean(stimL[(t>0) & (t<=0.5),:],axis=0)
    sortCol[1,:]  = np.nanmean(stimR[(t>0) & (t<=0.5),:],axis=0)
    sortCol[2,:]  = np.nanmean(beep[(t>0) & (t<=0.5),:],axis=0)
    sortCol[3,:]  = np.nanmean(moveL[(t>0) & (t<=0.5),:],axis=0)
    sortCol[4,:]  = np.nanmean(moveR[(t>0) & (t<=0.5),:],axis=0)
    sortCol[5,:]  = np.nanmean(feedbackP[(t>0) & (t<=0.5),:],axis=0)
    sortCol[6,:]  = np.nanmean(feedbackN[(t>0) & (t<=0.5),:],axis=0)
    sortCol[7,:]  = np.nanmean(lick[(t>0) & (t<=0.5),:],axis=0)
    
    if (not(focusOne is None)):
        for si in range(sortCol.shape[0]):
            sortCol[si,:] = sortCol[focusOne,:]
            sortBy[si,:]= sortBy[focusOne,:]

    
    if (len(sortBy)!=0) :        
        sortBy = np.delete(sortBy,Nans,axis=1)

        sorter = np.multiply(sortBy.astype(np.float64),sortCol)
        
        lastIndex = 0
        finalOrder = np.zeros(sortCol.shape[1])
        nonOrdered = 1
        for si in range(sorter.shape[0]):
            # Take all what is not sorted as NAN
            
      
            sorter[si,sortBy[si,:]==0] = np.nan

            sortedArray = np.sort(sorter[si,:])
            FirstNan = np.where(np.isnan(sortedArray))[0]
            if (len(FirstNan)==0):
                Ind = np.argsort(sorter[si,:])
                finalOrder[lastIndex:] = np.flip(Ind[:])
                nonOrdered = 0
                break;
            FirstNan = FirstNan[0]
            Ind = np.argsort(sorter[si,:])
            finalOrder[lastIndex:lastIndex+FirstNan] = np.flip(Ind[:FirstNan])
            lastIndex = lastIndex+FirstNan
            sorter[:,sortBy[si,:]==1] = np.nan
        
        if (nonOrdered):
            finalOrder[lastIndex:] = np.where(np.sum(sortBy,0)==0)[0]
        ind = finalOrder.astype(int)

        
    df = (t[1]-t[0])/2
    cmap = cm.seismic  
    cmap.set_bad('purple',1.)
    c = [-1,1]   
    fig0,ax = plt.subplots(1,2,figsize = (20,20))     
    fig0.subplots_adjust(wspace=0.1)               
    font = {'family' : 'Arial', 'size'   : 12}
    rc('font', **font)
    fig0.suptitle('Stimulus Locked Response')
    if (specificSort):
        ind = np.flip(np.argsort(sortCol[0,:])).astype(int)
    im = ax[0].imshow(stimL[:,ind].T,extent = [t[0]+df,t[-1]+df,1,stimL.shape[1]],aspect='auto',cmap = cmap)
    im.set_clim(c)
    if (specificSort):
        ind = np.flip(np.argsort(sortCol[1,:])).astype(int)
    im = ax[1].imshow(stimR[:,ind].T,extent = [t[0]+df,t[-1]+df,1,stimL.shape[1]],aspect='auto',cmap = cmap)
    im.set_clim(c)
    ax[0].set_xlabel('Time')
    ax[1].set_xlabel('Time')
    ax[0].set_ylabel('Neurons')
    ax[1].set_ylabel('Neurons')
    ax[0].set_title('Contra')
    ax[1].set_title('Ipsi')
    ax[0].set_ylim(0,stimL.shape[1])
    ax[1].set_ylim(0,stimL.shape[1])
    ax[0].vlines(0,0,stimL.shape[1],colors = 'k', linestyles='dashed')
    ax[1].vlines(0,0,stimL.shape[1],colors = 'k', linestyles='dashed')
    ax[1].axes.yaxis.set_visible(False)
    ax[0].set_xlim(-0.1,0.5)
    ax[1].set_xlim(-0.1,0.5)
    if (addLines): 
        ax[0].hlines(lines,t[0],t[-1],'k')
        ax[1].hlines(lines,t[0],t[-1],'k')
    
    
    fig1,ax = plt.subplots(1,2,figsize = (20,20))                    
    font = {'family' : 'Arial', 'size'   : 12}
    rc('font', **font)
    fig1.suptitle('Move Locked Response')
    if (specificSort):
        ind = np.flip(np.argsort(sortCol[2,:])).astype(int)
    im = ax[0].imshow(moveL[:,ind].T,extent = [t[0],t[-1],1,stimL.shape[1]],aspect='auto',cmap = cmap)
    im.set_clim(c)
    if (specificSort):
        ind = np.flip(np.argsort(sortCol[3,:])).astype(int)
    im = ax[1].imshow(moveR[:,ind].T,extent = [t[0],t[-1],1,stimL.shape[1]],aspect='auto',cmap = cmap)
    im.set_clim(c)
    ax[0].set_xlabel('Time')
    ax[1].set_xlabel('Time')
    ax[0].set_ylabel('Neurons')
    ax[1].set_ylabel('Neurons')
    ax[0].set_title('Move Contra')
    ax[1].set_title('Move Ipsi')
    ax[0].set_ylim(0,stimL.shape[1])
    ax[1].set_ylim(0,stimL.shape[1])
    ax[0].vlines(0,0,stimL.shape[1],colors = 'k',linestyles='dashed')
    ax[1].vlines(0,0,stimL.shape[1],colors = 'k',linestyles='dashed')
    ax[1].axes.yaxis.set_visible(False)
    fig1.subplots_adjust(wspace=0.1)
    ax[0].set_xlim(-0.1,0.5)
    ax[1].set_xlim(-0.1,0.5)
    if (addLines): 
        ax[0].hlines(lines,t[0],t[-1],'k')
        ax[1].hlines(lines,t[0],t[-1],'k')
    
    fig2,ax = plt.subplots(1,2,figsize = (20,20))                    
    font = {'family' : 'Arial', 'size'   : 12}
    rc('font', **font)   
    fig2.suptitle('Cue Locked Response')
    if (specificSort):
        ind = np.flip(np.argsort(sortCol[4,:])).astype(int)
    im = ax[0].imshow(beep[:,ind].T,extent = [t[0],t[-1],1,stimL.shape[1]],aspect='auto',cmap = cmap)
    im.set_clim(c)
    ax[0].set_xlabel('Time')
    ax[1].set_xlabel('Time')
    ax[0].set_ylabel('Neurons')
    ax[1].set_ylabel('Neurons')
    ax[0].set_title('Cue')
    ax[1].set_title('StimR')
    ax[1].set_visible(False)
    ax[0].set_ylim(0,stimL.shape[1])
    ax[1].set_ylim(0,stimL.shape[1])
    ax[0].vlines(0,0,stimL.shape[1],colors = 'k',linestyles='dashed')
    ax[1].vlines(0,0,stimL.shape[1],colors = 'k',linestyles='dashed')
    fig2.subplots_adjust(wspace=0.1)
    ax[0].set_xlim(-0.1,0.5)
    ax[1].set_xlim(-0.1,0.5)
    if (addLines): 
        ax[0].hlines(lines,t[0],t[-1],'k')
        ax[1].hlines(lines,t[0],t[-1],'k')
    
    fig3,ax = plt.subplots(1,2,figsize = (20,20))                    
    font = {'family' : 'Arial', 'size'   : 12}
    rc('font', **font) 
    fig3.suptitle('Feedback Locked Response')
    if (specificSort):
        ind = np.flip(np.argsort(sortCol[5,:])).astype(int)
    im = ax[0].imshow(feedbackP[:,ind].T,extent = [t[0],t[-1],1,stimL.shape[1]],aspect='auto',cmap = cmap)
    im.set_clim(c)
    if (specificSort):
        ind = np.flip(np.argsort(sortCol[6,:])).astype(int)
    im = ax[1].imshow(feedbackN[:,ind].T,extent = [t[0],t[-1],1,stimL.shape[1]],aspect='auto',cmap = cmap)
    im.set_clim(c)
    ax[0].set_xlabel('Time')
    ax[1].set_xlabel('Time')
    ax[0].set_ylabel('Neurons')
    ax[1].set_ylabel('Neurons')
    ax[0].set_title('Positive Feedback')
    ax[1].set_title('Negative Feedback')
    ax[0].set_ylim(0,stimL.shape[1])
    ax[1].set_ylim(0,stimL.shape[1])
    ax[0].vlines(0,0,stimL.shape[1],colors = 'k',linestyles='dashed')
    ax[1].vlines(0,0,stimL.shape[1],colors = 'k',linestyles='dashed')
    ax[1].axes.yaxis.set_visible(False)
    fig3.subplots_adjust(wspace=0.1)
    ax[0].set_xlim(-0.1,0.5)
    ax[1].set_xlim(-0.1,0.5)
    if (addLines): 
        ax[0].hlines(lines,t[0],t[-1],'k')
        ax[1].hlines(lines,t[0],t[-1],'k')
    
    fig4,ax = plt.subplots(1,2,figsize = (20,20))                    
    font = {'family' : 'Arial', 'size'   : 12}
    rc('font', **font)   
    fig4.suptitle('Lick Locked Response')
    if (specificSort):
        ind = np.flip(np.argsort(sortCol[4,:])).astype(int)
    im = ax[0].imshow(lick[:,ind].T,extent = [t[0],t[-1],1,stimL.shape[1]],aspect='auto',cmap = cmap)
    im.set_clim(c)
    ax[0].set_xlabel('Time')
    ax[1].set_xlabel('Time')
    ax[0].set_ylabel('Neurons')
    ax[1].set_ylabel('Neurons')
    ax[0].set_title('Lick')
    ax[1].set_title('StimR')
    ax[1].set_visible(False)
    ax[0].set_ylim(0,stimL.shape[1])
    ax[1].set_ylim(0,stimL.shape[1])
    ax[0].vlines(0,0,stimL.shape[1],colors = 'k',linestyles='dashed')
    ax[1].vlines(0,0,stimL.shape[1],colors = 'k',linestyles='dashed')
    fig4.subplots_adjust(wspace=0.1)
    ax[0].set_xlim(-0.1,0.5)
    ax[1].set_xlim(-0.1,0.5)
    if (addLines): 
        ax[0].hlines(lines,t[0],t[-1],'k')
        ax[1].hlines(lines,t[0],t[-1],'k')
    
    fig4.colorbar(im,ax=ax)
    return returnVal