# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 15:17:27 2022

@author: LABadmin
"""

def PredictionPerformanceSpecificPopulation(dataDir = 'D:\\Figures\\OneFunction_raw\\',flat = False ,reps = 500,iterations = [],testStat = 5, controlStat = None,controlState = 0,ephys = False, tups=None,contrast=None, split=0.5,preSplit=False,C=5,shuffle=False):
    tune = contrastTunerOneFunction()
    files = glob.glob(dataDir+ 'Data*.pickle', recursive=True)
    
    remFiles = []
    for fi in range(len(files)):
        if 'shuffle' in files[fi]:
            remFiles.append(fi)
    files = np.delete(files,remFiles)
    
    results = np.zeros((len(files),2,reps))*np.nan
    if (len(iterations)>0):
        files = np.array(files)[iterations]
    fi =0
    avgs0 = []
    avgs1 = []
    cons0 = []
    cons1 = []
    consI0 = []
    consI1 = []
    for fname in files:               
        sessionIdentifiers = re.findall('\d+',fname)
        sessionNumbers = int(sessionIdentifiers[0])
        if (ephys):
            probeNumbers = int(sessionIdentifiers[1])
        # print(fname+'\n')
        with open(fname, 'rb') as f:
            res = pickle.load(f) 
        X = res['X']        
        consCont = X[0,:]
        consIpsi = X[1,:]
        avgs = res['avgs']
        
        if (not(controlStat is None)):
            controlKeep = np.where(X[controlStat,:]==controlState)[0]
            X = X[:,controlKeep]
            consCont = consCont[controlKeep]
            consIpsi = consIpsi[controlKeep]
            avgs = avgs[controlKeep,:]
        # leave only trials where either no stim on both sides or contra lateral stim
        # cond = ((consIpsi==0) & (consCont>=0))
        # X = X[:,cond]
        # avgs = avgs[cond]
        # consCont = consCont[cond]
        # consIpsi = consIpsi[cond]
        
        if (not(contrast is None)):
            cond  = ((consCont==0))|(consCont==contrast)
            X = X[:,cond]
            avgs = avgs[cond]
            consCont = consCont[cond]
            consIpsi = consIpsi[cond]
        positive = np.where(X[-1,:]==1)[0]
        negative = np.where(X[-1,:]==0)[0]
        stimPresent = (consCont>0)
        N = avgs.shape[1]
        classifier = LogisticRegression(penalty='l2')
        
        if (not (tups is None)):
            if (not ephys):
                respNeurons = tups[tups[:,0]==sessionNumbers,1]
            else:
                respProbes = tups[tups[:,0]==sessionNumbers,1:]
                respNeurons = respProbes[respProbes[:,0]==probeNumbers,1]
            if (len(respNeurons)!=0):
                avgs = avgs[:,respNeurons]
            else:
                continue
        # Try fitting of negative previous feedback compared to positive
                    
                
        if (testStat!=-1):
            prevFeedback = X[testStat,:]
        else:
            prevFeedback = np.zeros(X.shape[1])
            oneInd = random.sample(range(avgs.shape[0]), avgs.shape[0]//2) 
            prevFeedback[oneInd] = 1
        # avgs_ = avgs[prevFeedback]
        # stimPresent_ = stimPresent[prevFeedback]
    
        if (shuffle):
            prevFeedback = np.random.permutation(prevFeedback)
        
        draw = range(N)
        
        print ('file: ' + fname)
        
        avgs0.append(avgs[prevFeedback==0])
        avgs1.append(avgs[prevFeedback==1])
        cons0.append(consCont[prevFeedback==0])
        cons1.append(consCont[prevFeedback==1])
        consI0.append(consIpsi[prevFeedback==0])
        consI1.append(consIpsi[prevFeedback==1])
        
        for rep in range(reps):
           
            if np.any(np.isnan(avgs)):
                continue;
            
            posInd = np.where(prevFeedback==1)[0]
            negInd = np.where(prevFeedback==0)[0]
            
            if (len(posInd)==0) or (len(negInd)==0):
                continue
            try:
                if (preSplit):
                    pos = avgs[posInd,:]
                    consPos = consCont[posInd]
                    stimPos = stimPresent[posInd]
                    neg = avgs[negInd,:]
                    consNeg = consCont[negInd]
                    stimNeg = stimPresent[negInd]
                    
                    X_train, X_test, y_train, y_test = train_test_split(range(pos.shape[0]), stimPos, test_size=split,stratify=stimPos)
                    logit = classifier.fit(pos[X_train,:],y_train)
                    scp = matthews_corrcoef(y_test,logit.predict(pos[np.array(X_test),:]))
                    results[fi,0,rep] = scp
                    
                    X_train, X_test, y_train, y_test = train_test_split(range(neg.shape[0]), stimNeg, test_size=split,stratify=stimNeg)
                    logit = classifier.fit(neg[X_train,:],y_train)
                    scn = matthews_corrcoef(y_test,logit.predict(neg[np.array(X_test),:]))
                    results[fi,1,rep] = scn
                else:
                    if (sum(stimPresent)<2 | (len(stimPresent) - sum(stimPresent))<2):
                        continue;
                    if (sum(prevFeedback)<2):
                        continue;
                    X_train, X_test, y_train, y_test = train_test_split(range(avgs.shape[0]), stimPresent, test_size=split,stratify=prevFeedback)
                    logit = classifier.fit(avgs[X_train,:],y_train)
                    testFeedbackStatus = prevFeedback[X_test].astype(bool)
                    if (sum(testFeedbackStatus==0)==0) or (sum(testFeedbackStatus==1)==0):
                        continue;
                    scp = matthews_corrcoef(y_test[testFeedbackStatus==0],logit.predict(avgs[np.array(X_test)[testFeedbackStatus==0],:]))
                    results[fi,0,rep] = scp
                    scn = matthews_corrcoef(y_test[testFeedbackStatus==1],logit.predict(avgs[np.array(X_test)[testFeedbackStatus==1],:]))
                    results[fi,1,rep] = scn
            except:
                continue;
                        
        fi+=1
    metadata = {'avgs0':avgs0 ,'avgs1':avgs1,'cons0': np.hstack(cons0) ,'cons1': np.hstack(cons1),'consI0': np.hstack(consI0) ,'consI1': np.hstack(consI1)} 
    return results,metadata

def GetPupilSize(allData,influence='feedback',fitTime = 0.500, iteration=0,plot=False):
    itRange = range(1,len(allData)+1)
    if (iteration!=0):
        itRange = iteration
    neuron_count=-1
    DataList = []
    InternalLists = []
    respNTot = 0
    
    MetadataList = []  
    pupilg1 = []
    pupilg2 = []
    
    
    for i in itRange: 
        
        print('Session' + str(i))
      
        
        stimStarts = allData[i]['stimT'][:,0]       
        choice = allData[i]['choice']              
        feedback = allData[i]['feedbackType']
        
        negTrials = np.where(feedback<=0)[0]
        posTrials = np.where(feedback==1)[0]
        changeFeedback = np.where(np.diff(feedback,axis=0)!=0)[0]+1 #the very first possible value is the second trial
        
        window = np.array([[-0.01, fitTime]])
        # wh,t = AlignStim(allData[i]['wheelVelocity'], allData[i]['times'], stimStarts, window)
        # wh = wh[((t<=fitTime) & (t>=0)),:,0]
        # wh = np.nanmean(np.abs(wh),0)
        # moveDuringStimInd = np.where(wh>0)[0]
        
        if not('pupil' in allData[i].keys()):
            continue;
        
        # Pupil
        # pupil_ds = AlignSignals(allData[i]['pupil'],allData[i]['pupilTimes'],allData[i]['calTimes'])
        pupil_ds = allData[i]['pupil']
        middle = np.nanmedian(pupil_ds,0)
        # pu,t = AlignStim(pupil_ds, times, stimStarts, window)     
        pu,t = AlignStim(pupil_ds, allData[i]['pupilTimes'], stimStarts, window)      
        bigTrials = pu>middle
        bigTrialsSum = np.sum(bigTrials[:,:,0],0)
        meanPupil = np.nanmedian(pu[t>=0,:,0],0)
        largePupil = (bigTrialsSum/len(t))>0.5
        smallPupil = (bigTrialsSum/len(t))<0.5       
        
        g1Name = 'Positive Feedback'
        g2Name = 'Negative Feedback'
        
        if (influence == 'feedback'):
            posBefore = posTrials + 1
            negBefore = negTrials + 1
            posBefore = posBefore[[posBefore<len(feedback)]]
            negBefore = negBefore[[negBefore<len(feedback)]]       
            
            
        
        # posBefore = np.setdiff1d(posBefore,moveDuringStimInd)
        # negBefore = np.setdiff1d(negBefore,moveDuringStimInd)
        
        pupilPos = meanPupil[posBefore]
        pupilNeg = meanPupil[negBefore]
        
        
        
        if (len(pupilPos)>0) & (len(pupilNeg)>0):
            pupilg1.append(np.nanmean(pupilPos))
            pupilg2.append(np.nanmean(pupilNeg))
        
        
    fig,ax = plt.subplots()
    t,p = sp.stats.ttest_rel(pupilg1,pupilg2)
    fig.suptitle('Pupil Size by ' + influence +' Across animals'+  '\nt: '+str(np.round(t,3))+' ,p: ' + str(np.round(p,3)) )
    ax.scatter(pupilg1,pupilg2,s=30,c=['k'],edgecolors='k')
    ax.plot(range(20,80),range(20,80),'--k')
    ax.set_xlabel(g1Name)
    ax.set_ylabel(g2Name)
    ax.set_ylim(20,80)
    ax.set_xlim(20,80)
    ax.set(aspect='equal')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    return pupilg1,pupilg2

def ShowLickingForState(allData,influence = 'feedback', iteration = 0):
    
    itRange = range(1,len(allData)+1)
    if (iteration!=0):
        itRange = iteration
    
    lickomnibusPos = []
    lickomnibusNeg = []
    overallLick = []
    overallSigPos = []
    overallSigNeg = []
    overallSigPosFb = []
    overallSigNegFb = []
    for i in itRange:
       
        sig = allData[i]['calTrace']
        times = allData[i]['calTimes']
        
        stimStarts = allData[i]['stimT'][:,0]
        conL = allData[i]['contrastLeft']
        conR = allData[i]['contrastRight']        
        conDiffs = conR-conL
         
        choice = allData[i]['choice']
        feedback = allData[i]['feedbackType']
        negTrials = np.where(feedback<=0)[0]
        posTrials = np.where(feedback==1)[0]
        changeFeedback = np.where(np.diff(feedback,axis=0)!=0)[0]+1 #the very first possible value is the second trial
        
        window = np.array([[-0.5, 0.5]])  
        
        # Pupil
        pupil_ds = AlignSignals(allData[i]['pupil'],allData[i]['pupilTimes'],allData[i]['calTimes'])
        middle = np.nanmedian(pupil_ds,0)
        pu,t = AlignStim(pupil_ds, times, stimStarts, np.array([[-0.001, 0.5]])  )        
        bigTrials = pu>middle
        bigTrialsSum = np.sum(bigTrials[:,:,0],0)
        meanPupil = np.nanmean(pu[:,:,0],0)
        largePupil = (bigTrialsSum/len(t))>0.5
        smallPupil = (bigTrialsSum/len(t))<0.5
        
        saveDir = 'C:\\TaskProject\\Figures\\States\\'+influence+'_raw\\'
        g1Name = 'Positive Feedback'
        g2Name = 'Negative Feedback'
        if (influence == 'feedback'):
            posBefore = posTrials + 1
            negBefore = negTrials + 1
            posBefore = posBefore[[posBefore<len(feedback)]]
            negBefore = negBefore[[negBefore<len(feedback)]]
            
        if (influence == 'feedbackPos'):
            posBefore = posTrials + 1
            negBefore = negTrials + 1
            posBefore = posBefore[[posBefore<len(feedback)]]
            negBefore = negBefore[[negBefore<len(feedback)]]
            
            posBefore = np.intersect1d(posTrials,posBefore)
            negBefore = np.intersect1d(posTrials,negBefore)
        
        if (influence == 'feedbackNeg'):
            posBefore = posTrials + 1
            negBefore = negTrials + 1
            posBefore = posBefore[[posBefore<len(feedback)]]
            negBefore = negBefore[[negBefore<len(feedback)]]
            
            posBefore = np.intersect1d(negTrials,posBefore)
            negBefore = np.intersect1d(negTrials,negBefore)
            
        if (influence == 'correctness'): 
            g1Name = 'correct'
            g2Name = 'incorrect'
            # fazedOutSum
            # noGoCount = np.zeros(choice.shape)
            # wrongNogo= ((choice==0) & (feedback==0))
            # for c in range(len(wrongNogo)-1):
            #     if (wrongNogo[c,:]):
            #         noGoCount[c,:] +=1
            #         noGoCount[c+1,:] = noGoCount[c,:]
            #     else:
            #         noGoCount[c,:] = 0
            # posBefore = np.where(noGoCount<2)[0]
            # negBefore = np.where(noGoCount>=2)[0]
            move = np.where(choice!=0)[0]
            posBefore = posTrials
            negBefore = negTrials
            
            posBefore = np.intersect1d(move,posBefore)
            negBefore = np.intersect1d(move,negBefore)
            
        if (influence == 'move'): 
            g1Name = 'Go'
            g2Name = 'No Go'
            # fazedOutSum
            
            posBefore = np.where(choice!=0)[0]
            negBefore = np.where(choice==0)[0]
            
        if (influence == 'direction'): 
            g1Name = 'Contra'
            g2Name = 'Ipsi'
            # fazedOutSum
            
            posBefore = np.where(choice>0)[0]
            negBefore = np.where(choice<0)[0]
            
        if (influence == 'pupil'):   
            g1Name = 'Large Pupil'
            g2Name = 'Small pupil'
            posBefore = np.where(largePupil)[0]
            negBefore = np.where(smallPupil)[0]
        
        pupilPos = meanPupil[posBefore]
        pupilNeg = meanPupil[negBefore]
            
          
            
        delays = allData[i]['planeDelays']
        planes = allData[i]['planes']   
        
        
        lickraw = allData[i]['lick_raw'][:,0]
        lickTimes = allData[i]['wheelTimes']
        lickRate = getLickRate(lickraw,lickTimes)
        
        
        lckStim,t = AlignStim(lickRate, lickTimes, stimStarts, window)            
        lckFeedback,t = AlignStim(lickRate, lickTimes, allData[i]['feedbackTimes'][:,0], window)         
        
        
        testTime = np.where((t>0.01) & (t<=0.5))[0]
        preTime = np.where((t>-0.5) & (t<=0))[0]
        
        # Stim
        g1 = lckStim[:,posBefore,0]
        g1post = np.mean(g1[testTime],0)
        g1pre = np.mean(g1[preTime],0)
        g2= lckStim[:,negBefore,0]
        g2post = np.mean(g2[testTime],0)
        g2pre = np.mean(g2[preTime],0)
        tval,p = sp.stats.ttest_ind(g1post,g2post)
        # Feedback
        g1f = lckFeedback[:,posBefore,0]
        g1postf = np.mean(g1f[testTime],0)
        g1pref = np.mean(g1f[preTime],0)
        g2f= lckFeedback[:,negBefore,0]
        g2postf = np.mean(g2f[testTime],0)
        g2pref = np.mean(g2f[preTime],0)
        tval,p = sp.stats.ttest_ind(g1post,g2post)
        
        
        
        y1 = np.nanmean(lckStim[:,posBefore,0],1)
        err1 = np.nanstd(lckStim[:,posBefore,0],1)/np.sqrt(len(posBefore))
        
        y2 = np.nanmean(lckStim[:,negBefore,0],1)
        err2 = np.nanstd(lckStim[:,negBefore,0],1)/np.sqrt(len(negBefore))
        
        overallSigPos.append(lckStim[:,posBefore,0])
        overallSigPosFb.append(lckFeedback[:,posBefore,0])
        y1 = np.nanmean(lckFeedback[:,posBefore-1,0],1)
        err1 = np.nanstd(lckFeedback[:,posBefore-1,0],1)/np.sqrt(len(posBefore))
            
        y2 = np.nanmean(lckFeedback[:,negBefore-1,0],1)
        err2 = np.nanstd(lckFeedback[:,negBefore-1,0],1)/np.sqrt(len(negBefore))
        
        overallSigNeg.append(lckStim[:,negBefore,0])
        overallSigNegFb.append(lckFeedback[:,negBefore,0])
        overallLick.append((g1pre,g1post,g2pre,g2post,  g1pref,g1postf,g2pref,g2postf))   
        
        
    lckStimPos = np.hstack(overallSigPos)
    lckFeedbackPos = np.hstack(overallSigPosFb)
    lckStimNeg = np.hstack(overallSigNeg)
    lckFeedbackNeg= np.hstack(overallSigNegFb)
    
    fig,ax = plt.subplots(1,2)
    fig.suptitle('Overall')
    y1 = np.nanmean(lckStimPos,1)
    err1 = np.nanstd(lckStimPos,1)/np.sqrt(lckStimPos.shape[1])
    ax[1].plot(t,y1,'b')
    ax[1].fill_between(t,y1-err1,y1+err1,alpha=0.5,facecolor='blue')        
    y2 = np.nanmean(lckStimNeg,1)
    err2 = np.nanstd(lckStimNeg,1)/np.sqrt(lckStimNeg.shape[1])
    ax[1].plot(t,y2,'r')
    ax[1].fill_between(t,y2-err1,y2+err2,alpha=0.5,facecolor='red')
    ax[1].set_title('Stim Locked licking')
    
    y1 = np.nanmean(lckFeedbackPos,1)
    err1 = np.nanstd(lckFeedbackPos,1)/np.sqrt(lckFeedbackPos.shape[1])
    ax[0].plot(t,y1,'b')
    ax[0].fill_between(t,y1-err1,y1+err1,alpha=0.5,facecolor='blue')        
    y2 = np.nanmean(lckFeedbackNeg,1)
    err2 = np.nanstd(lckFeedbackNeg,1)/np.sqrt(lckFeedbackNeg.shape[1])
    ax[0].plot(t,y2,'r')
    ax[0].fill_between(t,y2-err1,y2+err2,alpha=0.5,facecolor='red')
    ax[0].set_title('Feedback Locked licking')
    ax[1].set_xlim(ax[0].get_xlim())
    ax[0].spines['right'].set_visible(False)
    ax[0].spines['top'].set_visible(False)  
    ax[1].spines['right'].set_visible(False)
    ax[1].spines['top'].set_visible(False) 
    plt.tight_layout()
    return overallLick

def GetResponseProgression(allData,specificNeurons=[],fitTime = 0.500,iteration = 0):
    itRange = range(1,len(allData)+1)
    if (iteration!=0):
        itRange = iteration   
    
    Progs = []
    Progs_cons = []
    pup = []
    fb = []
    for i in itRange:
       
        sig = allData[i]['calTrace']
        times = allData[i]['calTimes']
        
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
        ng = choice!=0
        ngs = np.split(ng,np.where(ng)[0])
        feedback = allData[i]['feedbackType']     
        
        window = np.array([[-1, 1]])  
           
        delays = allData[i]['planeDelays']
        planes = allData[i]['planes'] 
        
         # Pupil
        windowp = np.array([[-0.5, 0.5]])  
        pupil_ds = AlignSignals(allData[i]['pupil'],allData[i]['pupilTimes'],allData[i]['calTimes'])
        middle = np.nanmedian(pupil_ds,0)
        pu,t = AlignStim(pupil_ds, times, stimStarts, windowp)        
        bigTrials = pu>middle
        meanPupil = np.nanmean(pu[:,:,0],0)
         
        
        # Inds for contrast or 0/0
        testContrasts_c = np.where((contra>0) | ((contra==0) & (ipsi==0)))[0]
        testContrasts_i = np.where((ipsi>0) | ((contra==0) & (ipsi==0)))[0]
        
        ca_raw,t = GetCalciumAligned(sig, times, stimStarts, window,planes,delays)
        ca_raw = ca_raw/np.max(np.max(ca_raw,1),0)
        ca = ca_raw - np.tile(np.mean(ca_raw[t<=0,:,:],axis=0),(ca_raw.shape[0],1,1))
        testTime = np.where((t>0) & (t<fitTime))[0]
        
        avgs = np.nanmean(ca[testTime,:,:],0)       
        
        # Get average for each contrast
        conAvgs = np.zeros(len(uniqueConContra))
        conSd = np.zeros(len(uniqueConContra))
        for ci in range(len(uniqueConContra)):
            Inds = np.intersect1d(np.where(contra==uniqueConContra[ci])[0],testContrasts_c)
            conAvgs[ci] = np.nanmean(avgs[Inds])
            conSd[ci] = np.nanstd(avgs[Inds])
        
        deltaProgression = np.zeros((len(stimStarts),ca.shape[2]))
        deltaProgression_cons = np.zeros((len(uniqueConContra),len(stimStarts),ca.shape[2]))
        for trial in range(len(stimStarts)):
            if trial in testContrasts_c:
                cInd = np.where(contra[trial]==uniqueConContra)[0]
                deltaProgression[trial] = (avgs[trial,:]-conAvgs[cInd])/conSd[cInd]
                deltaProgression_cons[cInd,trial] = (avgs[trial,:]-conAvgs[cInd])/conSd[cInd]
        
        Progs.append(deltaProgression)
        Progs_cons.append(deltaProgression_cons)
        pup.append(meanPupil)
        fb.append(feedback)
        
    return Progs,Progs_cons,pup,fb