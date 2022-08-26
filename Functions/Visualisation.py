# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 10:14:33 2022

@author: LABadmin
"""

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
    
    
        
    
    
    # BL subtract
    # BL = (np.nanmean(stimL[t<0,:],axis=0) + np.nanmean(stimR[t<0,:],axis=0))/2
    # # stimL = stimL - np.nanmean(stimL[t<0,:],axis=0)
    # # stimR = stimR - np.nanmean(stimR[t<0,:],axis=0)
    # # beep = beep - np.nanmean(beep[t<0,:],axis=0)
    # # moveL = moveL - np.nanmean(moveL[t<0,:],axis=0)
    # # moveR = moveR - np.nanmean(moveR[t<0,:],axis=0)
    # # feedbackP = feedbackP - np.nanmean(feedbackP[t<0,:],axis=0)
    # # feedbackN = feedbackN - np.nanmean(feedbackN[t<0,:],axis=0)
    # stimL = stimL - BL
    # stimR = stimR - BL
    # beep = beep - BL
    # moveL = moveL - BL
    # moveR = moveR - BL
    # feedbackP = feedbackP - BL
    # feedbackN = feedbackN - BL
    # lick = lick - BL
    
    # BL = (np.nanmean(stimL[t<0,:],axis=0) + np.nanmean(stimR[t<0,:],axis=0))/2
    # stimL = stimL - np.nanmean(stimL[t<0,:],axis=0)
    # stimR = stimR - np.nanmean(stimR[t<0,:],axis=0)
    # beep = beep - np.nanmean(beep[t<0,:],axis=0)
    # moveL = moveL - np.nanmean(moveL[t<0,:],axis=0)
    # moveR = moveR - np.nanmean(moveR[t<0,:],axis=0)
    # feedbackP = feedbackP - np.nanmean(feedbackP[t<0,:],axis=0)
    # feedbackN = feedbackN - np.nanmean(feedbackN[t<0,:],axis=0)
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
    # cmin= min(np.nanmin(stimL),np.nanmin(stimR),np.nanmin(beep),np.nanmin(moveL),np.nanmin(moveR),np.nanmin(feedbackP),np.nanmin(feedbackN))
    # cmax = max(np.nanmax(stimL),np.nanmax(stimR),np.nanmax(beep),np.nanmax(moveL),np.nanmax(moveR),np.nanmax(feedbackP),np.nanmax(feedbackN))
    # cc = max(abs(cmin),abs(cmax))-0.2
    # c = [-cc,cc]
    c = [-4,4]
    depthInds = np.array([0,stimL.shape[1]])
    ind = np.flip(np.lexsort(sortCol))
    # if (len(sortBy)>0):
    #     sortBy = np.delete(sortBy,Nans,axis=1)
    #     ind =np.lexsort(sortBy.astype(float))
    #     # ind = np.lexsort(np.flip(sortBy,axis=0).astype(float))
    #     if (not depth is None):
    #         depth = np.delete(depth,Nans,axis=1)
    #         ind1 = np.where(depth<=400)[1]
    #         ind2 = np.where((depth>400) & (depth<=700))[1]
    #         ind3 = np.where(depth>700)[1]
    #         depth[0,ind1] = 1
    #         depth[0,ind2] = 2
    #         depth[0,ind3] = 3
    #         sortBy = np.vstack((sortBy,depth))
    #         ind =np.lexsort(sortBy.astype(float))
    #         # get first and last instances of each depth
    #         depthInds = np.zeros((2,3))
    #         depthInds[:,0] = [0,np.where(ind==1)[0][-1]+1]
    #         depthInds[:,1] = [np.where(ind==2)[0][0],np.where(ind==2)[0][-1]+1]
    #         depthInds[:,2] = [np.where(ind==3)[0][0],np.where(ind==3)[0][-1]+1]
    
    if (addLines):        
        lines = np.zeros(7)
        for i in range(7):
            sb = np.where(sortBy==i)[1]
            _,inInd1,inInd2 = np.intersect1d(ind,sb,return_indices = True)
            lines[i] = inInd1[-1]
            
    
    # if (len(sortBy)>0):
    #     ind = np.flip(np.lexsort(np.flip(sortBy,axis=0).astype(int)))
    
    if (len(sortBy)!=0) :        
        sortBy = np.delete(sortBy,Nans,axis=1)
       
        # sorter[:,allZeros] = np.nan
        # sortPower[sortBy==0] = 0
        sorter = np.multiply(sortBy.astype(np.float64),sortCol)
        
        lastIndex = 0
        finalOrder = np.zeros(sortCol.shape[1])
        for si in range(sorter.shape[0]):
            # Take all what is not sorted as NAN
            
            # sorterTmp = sorter.copy()
            # sorterTmp[sortBy[si,:]==0] = np.nan
            sorter[si,sortBy[si,:]==0] = np.nan
            # sorter[si,:] = sortBy.astype(np.float64)[si,:]*sortCol[si,:]  
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