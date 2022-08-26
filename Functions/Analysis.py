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