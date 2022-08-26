# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 09:18:12 2022

@author: LABadmin
"""

def GetChoiPerAnimal(AllData,iteration=0, plot = 0):
   itRange = range(1,len(AllData)+1)
   if (iteration!=0):
       itRange = iteration 
   probs = np.zeros((1,6))  
   totPerfs = []
   totCons = []
   for i in itRange:
       
        d = AllData[i]
        t = d['calTimes']
        feedback = d['feedbackType']
        conL = d['contrastLeft']
        conR = d['contrastRight']
        conDiff = conR-conL
        uniqueDiffs= np.unique(conDiff)
        choice = d['choice']
        
        # if i>=11:
        #     choice = -choice
        perfs = np.zeros((3,len(uniqueDiffs)))
        
        #remove phased out trials
        disengagedInds = []
        noGoInds = np.where(choice==0)[0]
           
        for ng in noGoInds:
            if (ng>2) & ((ng-1) in noGoInds) & ((ng-2) in noGoInds):
                disengagedInds.append(ng)
        
        feedback = np.delete(feedback,disengagedInds)
        conDiff = np.delete(conDiff,disengagedInds)
        choice = np.delete(choice,disengagedInds)
        
        
        rightCons = conDiff[choice==1]
        noGoCons = conDiff[choice==0]
        leftCons = conDiff[choice==-1]
       
            
        
        for c in range(len(uniqueDiffs)):
            con = uniqueDiffs[c]                
            conChoice= choice[(conDiff==con)]
                                            
            perfs[0,c] = np.sum(conChoice==1)/len(conChoice)
            perfs[1,c] = np.sum(conChoice==0)/len(conChoice)
            perfs[2,c] = np.sum(conChoice==-1)/len(conChoice)
            
        totPerfs.append(perfs)
        totCons.append(uniqueDiffs)        
        
   if (plot):
        choiceCons = [-1,-0.5,0,0.5,1]
        
        overall= np.zeros((3,len(choiceCons),len(totPerfs)))
        for i in range(len(totPerfs)):
            cs = totCons[i]
            pf = totPerfs[i]
            
            for c in range(len(choiceCons)):
                if (sum(cs==choiceCons[c])==0):
                    overall[:,c,i] = np.nan
                    continue;
                overall[:,c,i] = pf[:,cs==choiceCons[c]][:,0]
        f,ax = plt.subplots(1)
        side = np.tile(np.array(['Right','NoGo','Left']).T,(16,5,1)).T
        contrast = np.moveaxis(np.tile(np.array(choiceCons),(3,16,1)),2,1)
        df = pd.DataFrame({'side':side.flatten(), 'contrast':contrast.flatten(),
                  'y':overall.flatten()})
        sns.lineplot(data = df,x='contrast',y='y',hue='side',legend = False,ax = ax,palette=['navy','gold','green'])
        ax.set_xticks(range(len(choiceCons)))
        ax.set_xticklabels(choiceCons)
        ax.set_ylabel('Correct (%)')
        ax.set_xlabel('Contrast')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_aspect('equal', adjustable='box') 
        ax.set_xlim(-1,1)
        ax.set_ylim(0,1)
        ax.set_yticks([0,0.5,1])
        ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
   return totPerfs

def GetReactionTime(AllData,iteration=0, plot = 0, stim=False,ephys=False):
   itRange = range(1,len(AllData)+1)
   if (iteration!=0):
       itRange = iteration 
   probs = np.zeros((1,6))  
   totPerfs = []
   totCons = []
   for i in itRange:
       
        d = AllData[i]
        if (ephys):
            t = d['times']
            wheelT = t
        else:
            t = d['calTimes']
            wheelT = d['wheelTimes']
        feedback = d['feedbackType']
        choice = d['choice']
        correctTrials = np.where((feedback==1) &(choice!=0))[0]
        feedback = feedback[correctTrials]
        choice = choice[correctTrials]
        conL = d['contrastLeft'][correctTrials]
        conR = d['contrastRight'][correctTrials]
        conDiff = conR-conL
        uniqueDiffs= np.unique(conDiff)    
        uniqueDiffs = np.sort(np.union1d(uniqueDiffs,[0]))
        wheelV = d['wheelVelocity']
        # wheel_ds = AlignSignals(d['wheelVelocity'][::10,:],d['wheelTimes'][::10,:],d['calTimes'])        
        # wheelMoveTime,_ =CreateMovementOnsetSignal(wheel_ds, d['calTimes'])
        cue = d['goCueTimes'][:,0][correctTrials]   
        if (stim):
           cue = d['stimT'][:,0][correctTrials] 
        
        rts = np.zeros(len(cue))
        
        
        for tt in range(len(cue)):    
            if (conDiff[tt]==0):
                rts[tt] = np.nan
                continue
            
            tCue = cue[tt]
            
            movTime = np.where((wheelT>tCue)&(wheelT<tCue+2))[0]
            
            cueVel = wheelV[movTime,0]
            
            if (choice[tt]==-1):
                cueVel = - cueVel
            
            pks, _= signal.find_peaks(cueVel,prominence=np.nanmean(cueVel)+2*np.std(cueVel) ,distance = 1)
    
    
            if (len(pks)>0):                
                res = signal.peak_widths(cueVel, pks, rel_height=0.001)
                res = np.floor(res[2]).astype(int)[0]
                rts[tt] = wheelT[movTime[res]]- wheelT[movTime[0]] 
            else:
                rts[tt]=np.nan
                continue
            
            
        
        conRts = np.zeros((150,len(uniqueDiffs)))*np.nan
        
        for c in range(len(uniqueDiffs)):            
            con = uniqueDiffs[c]  
            conRt = rts[conDiff[:,0]==con]
            conRts[:len(conRt),c] = conRt
            
        totPerfs.append(conRts)
        totCons.append(uniqueDiffs)
       
   if (plot):
         choiceCons = [-1,-0.5,0,0.5,1]
        
         overall= np.zeros((len(choiceCons),len(totPerfs)))
         for i in range(len(totPerfs)):
             cs = totCons[i]
             pf = totPerfs[i]
            
             for c in range(len(choiceCons)):
                 if (sum(cs==choiceCons[c])==0):
                     overall[c,i] = np.nan
                     continue;
                 overall[c,i] = np.nanmean(pf[:,cs==choiceCons[c]][:,0])
         f,ax = plt.subplots(1)         
         contrast = np.tile(choiceCons,(16,1)).T
         df = pd.DataFrame({'contrast':contrast.flatten(),
                   'y':(overall*1000).flatten()})
         sns.pointplot(data = df,x='contrast',y='y',legend = False,ax = ax)
         ax.set_xticks(range(len(choiceCons)))
         ax.set_xticklabels(choiceCons)
         ax.set_ylabel('RT(ms)')
         ax.set_xlabel('Contrast')
         ax.spines['right'].set_visible(False)
         ax.spines['top'].set_visible(False)
         # ax.set_xlim(-1,1)
         ax.set_yticks([500,750,1000])
         
         ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
   return totPerfs