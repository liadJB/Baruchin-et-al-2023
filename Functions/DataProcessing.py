# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 11:53:02 2022

@author: LABadmin
"""

def AlignStim(signal, time, eventTimes, window,timeUnit=1,timeLimit=1):
    aligned = [];
    t = [];

    dt = np.median(np.diff(time,axis=0))
    if (timeUnit==1):
        w = np.rint(window / dt).astype(int)
    else:
        w = window.astype(int)
    maxDur = signal.shape[0]
    if (window.shape[0] == 1): # constant window
        mini = np.min(w[:,0]);
        maxi = np.max(w[:,1]);
        tmp = np.array(range(mini,maxi));
        w = np.tile(w,((eventTimes.shape[0],1)))
    else:
        if (window.shape[0] != eventTimes.shape[0]):
            print('No. events and windows have to be the same!')
            return 
        else:
            mini = np.min(w[:,0]);
            maxi = np.max(w[:,1]);
            tmp = range(mini,maxi); 
    t = tmp * dt;

    aligned = np.zeros((t.shape[0],eventTimes.shape[0],signal.shape[1]))

    for ev in range(eventTimes.shape[0]):
    #     evInd = find(time > eventTimes(ev), 1);
        
        wst = w[ev,0]
        wet = w[ev,1]
        
        evInd = np.where(time>=eventTimes[ev])[0]
        if (len(evInd)==0): 
            continue
        else :
            # None
            # if dist is bigger than one second stop
            if (np.any((time[evInd[0]]-eventTimes[ev])>timeLimit)):
                continue
            
        st = evInd[0]+ wst #get start
        et = evInd[0] + wet  #get end        
        
        alignRange = np.array(range(np.where(tmp==wst)[0][0],np.where(tmp==wet-1)[0][0]+1))
        
       
        sigRange = np.array(range(st,et))
       
        valid = np.where((sigRange>=0) & (sigRange<maxDur))[0]
      
        aligned[alignRange[valid],ev,:] = signal[sigRange[valid],:];

    return aligned, t

def GetRaster(spikeTimes, eventTimes, window,dt=1/1000,cluster = []):
    ############################## Previous
    # aligned = [];
    # t = [];  
   
    # w = np.rint(window / dt).astype(int)
    
    # maxDur =np.nanmax(eventTimes)+window[1]
    
    # mini = np.min(w[0]);
    # maxi = np.max(w[1]);
    # tmp = np.array(range(mini,maxi));
    # w = np.tile(w,((eventTimes.shape[0],1)))    
    
    # t = tmp * dt;

    # aligned = np.zeros((t.shape[0],eventTimes.shape[0]))*np.nan
    # spikeTimes = spikeTimes[spikeTimes[:,0]<=maxDur,:]
    
    # Ind = []
    # if (len(cluster)!=0):
    #     for c in cluster:
    #         Ind.append(np.where(spikeTimes[:,1]==c)[0])
    #         spikeTimes = spikeTimes[np.array(Ind)[0,:],:]
    # for ev in range(eventTimes.shape[0]):
    # #     evInd = find(time > eventTimes(ev), 1); 
        
    #     wst = w[ev,0]
    #     wet = w[ev,1]
        
    #     st = eventTimes[ev]+ window[0] #get start
    #     et = eventTimes[ev] + window[1]  #get end        
        
    #     if (np.isnan(st)):
    #         continue
    #     alignRange = np.array(range(np.where(tmp==wst)[0][0],np.where(tmp==wet-1)[0][0]+1))
        
       
    #     sigRange = np.array(np.arange(st,et,dt))
       
    #     valid = np.where((sigRange>=0) & (sigRange<maxDur))[0]
        
        
    #     sigValid = sigRange[valid]
        
    #     spNow = spikeTimes[((spikeTimes[:,0]>=sigValid[0]) & (spikeTimes[:,0]<=sigValid[-1])),: ]
    #     spt = [i for i, e in enumerate(sigValid) if np.round(e,3) in np.round(spNow,3)] # probably a bug, adds more spikes than were found
        
    #     aligned[alignRange[spt],ev] = 1;
    #     aligned[np.isnan(aligned)] = 0
        ############################## Previous
        ###############################CURRENT
    aligned = [];
    t = [];  
   
    w = np.rint(window / dt).astype(int)
    
    maxDur =np.nanmax(eventTimes)+window[1]
    
    mini = np.min(w[0]);
    maxi = np.max(w[1]);
    tmp = np.array(range(mini,maxi));
    w = np.tile(w,((eventTimes.shape[0],1)))    
    
    t = tmp * dt;

    
    
    Ind = []
    if (len(cluster)!=0):
        for c in cluster:
            Ind.append(np.where(spikeTimes[:,1]==c)[0])
            spikeTimes = spikeTimes[np.array(Ind)[0,:],:]
            
    # convert clusterId to ordinal numbers
    uniqueClusters = np.unique(spikeTimes[:,1])    
    for c in range(len(uniqueClusters)):
        cl = uniqueClusters[c]
        spikeTimes[spikeTimes[:,1]==cl,1] = c
        
    aligned = np.empty((t.shape[0],eventTimes.shape[0],len(uniqueClusters)),dtype=bool)
    spikeTimes = spikeTimes[spikeTimes[:,0]<=maxDur,:]
        
    
    for ev in range(eventTimes.shape[0]):
    #     evInd = find(time > eventTimes(ev), 1); 
        
        wst = w[ev,0]
        wet = w[ev,1]
        if (np.isnan(eventTimes[ev])):
            continue;
        st = eventTimes[ev]+ window[0] #get start
        et = eventTimes[ev] + window[1]  #get end       
        
        winTmp = np.arange(st,et,dt)
        win = np.zeros((len(winTmp),len(uniqueClusters)))
        
        if (np.isnan(st)):
            continue
        alignRange = np.array(range(np.where(tmp==wst)[0][0],np.where(tmp==wet-1)[0][0]+1))
        
       
        sigRange = np.array(np.arange(st,et,dt))
       
        valid = np.where((sigRange>=0) & (sigRange<maxDur))[0]
        
        
        sigValid = sigRange[valid]
        
        spNow = spikeTimes[((spikeTimes[:,0]>=sigValid[0]) & (spikeTimes[:,0]<=sigValid[-1])),: ]
        spNowOrdinal = spNow        
        spNowOrdinal[:,0]/=dt
        spNowOrdinal[:,0]-=st/dt
        spNowOrdinal = spNowOrdinal.astype(int)
        win[spNowOrdinal[:,0],spNowOrdinal[:,1]] = 1         
        aligned[:,ev,:] = win[0:aligned.shape[0],:];
        ###################################
    return aligned.astype(bool), t

import scipy.ndimage

def halfgaussian_kernel1d(sigma, radius):
    """
    Computes a 1-D Half-Gaussian convolution kernel.
    """
    sigma2 = sigma * sigma
    x = np.arange(0, radius+1)
    phi_x = np.exp(-0.5 / sigma2 * x ** 2)
    phi_x = phi_x / phi_x.sum()

    return phi_x

def halfgaussian_filter1d(input, sigma, axis=-1, output=None,
                      mode="constant", cval=0.0, truncate=4.0):
    """
    Convolves a 1-D Half-Gaussian convolution kernel.
    """
    sd = float(sigma)
    # make the radius of the filter equal to truncate standard deviations
    lw = int(truncate * sd + 0.5)
    weights = halfgaussian_kernel1d(sigma, lw)
    origin = -lw // 2
    return scipy.ndimage.convolve1d(input, weights, axis, output, mode, cval, origin)

def GetPSTH(raster,w=5,halfGauss=False, sigma = 25 , flat=True):
    N = raster.shape[1]
    if (flat):        
        raster = pd.DataFrame(np.sum(raster.astype(int),1))
    else:
        raster = pd.DataFrame(raster)
    windows = raster.rolling(w,min_periods=1)
    if (halfGauss):
        return halfgaussian_filter1d((np.array(windows.sum())/(N*w/1000)),sigma,axis=0)
    return (np.array(windows.sum())/(N*w/1000))
    
def GetCalciumAligned(signal, time, eventTimes, window,planes,delays):
    aligned = []
    run = 0
    for p in range(delays.shape[1]):
       aligned_tmp, t = AlignStim(signal[:,np.where(planes==p+1)[0]], time + delays[0,p], eventTimes, window)
       if (run==0):
           aligned = aligned_tmp
           run+=1
       else:    
           aligned = np.concatenate((aligned,aligned_tmp),axis=2)
        
    return np.array(aligned),t


# Get closest approximation in time of a signal to a downsample
def AlignSignals(sig1,ts1,ts2,bigger= True):
    sigF = np.zeros((ts2.shape[0],sig1.shape[1]))
    # get the difference between each time signal
    df1 = np.nanmedian(np.diff(ts1,axis=0),axis=0)
    df2 = np.nanmedian(np.diff(ts2,axis=0),axis=0)
    ratio = int(np.ceil(df2/(2*df1))) # divide the slow time by the fast
    sig1 = signal.medfilt(sig1,(2*ratio-1,1)) # make sure all timepoints are windowed
    # if (len(ts1)>len(ts2)):
        # downsample
    for t in range(len(ts2)):
        if (bigger):
            t_ind = np.where(ts1>=ts2[t])[0][0]
        else:
            t_ind = np.argmin(abs(ts2[t]-ts1))
        sigF[t,:] = sig1[t_ind,:]
    return sigF

def CreateMovementOnsetSignal(velocity, ts, toleranceDuration = 0.1, movDist = 0.3):
    
    dt = ts[1,0]-ts[0,0]
    secs = np.ceil(1/dt)
    
    
    # velocity = velocity/np.nanmax(velocity)
    v_raw = velocity
    velocity= np.abs(velocity)
    velDiff = np.diff(velocity,axis=0)
    pks, _= signal.find_peaks(velocity[:,0],prominence=np.nanmean(velocity)+2*np.std(velocity) ,distance = secs*movDist)
    
    res = signal.peak_widths(velocity[:,0], pks, rel_height=0.001)
    
    movStartActual = np.floor(res[2]).astype(int) # the left of 0.1 of the peak
    
    zeros = np.where((velocity>=0) & (velocity<=0.1))[0]
    zerosSet = frozenset(zeros)
    zerosTs = ts[zeros,0]
    
    # movStartTs = ts[int(np.round(movStartActual)),0]
    remInds = np.zeros(len(movStartActual))
    
    for i in range(len(movStartActual)):
        curr = movStartActual[i]
        ind = 1
        closestZero = np.where(zeros<=curr)[0][-1]
        # make sure that there is not movement befor
        # while (ind<len(zeros)) & np.any(velocity[zeros[closestZero]-int(np.round(toleranceDuration*secs)):zeros[closestZero]]>0.1) & (np.any(velDiff[zeros[closestZero]-int(np.round(toleranceDuration*secs)):zeros[closestZero]]!=0)):
             
        #       closestZero = np.where(zeros<=curr)[0]
        #       if (len(closestZero)<2):
        #           break;
        #       else:
        #           closestZero = closestZero[-1-ind]
        #           ind+=1
        movStartActual[i] = zeros[closestZero]
        if (not(set(range(int(np.floor(movStartActual[i]-secs*movDist)),movStartActual[i]-1)).issubset((zerosSet)))):
            remInds[i] = 1
        # if (not(range(int(np.floor(movStartActual[i]-secs*movDist)),movStartActual[i]-1) in (np.sort(zeros)))):
            # remInds[i] = 1
     
    
   
         
        
        
        # make sure that there is not movement befor
        # while (ind<len(zeros)) & np.any(velocity[zeros[closestZero]-int(np.round(toleranceDuration*secs)):zeros[closestZero]]>0.1) & (np.any(velDiff[zeros[closestZero]-int(np.round(toleranceDuration*secs)):zeros[closestZero]]!=0)):
             
        #       closestZero = np.where(zeros<=curr)[0]
        #       if (len(closestZero)<2):
        #           break;
        #       else:
        #           closestZero = closestZero[-1-ind]
        #           ind+=1
    
    removeIndex = np.where(remInds==1)[0]    
    if (len(removeIndex)!=0):
        movStartActual = np.delete(movStartActual,removeIndex,axis=0)
        pks = np.delete(pks,removeIndex,axis=0)
    movStartActual = movStartActual.astype(int)
    
    # 
    direction = np.zeros((len(movStartActual),1))
    direction[v_raw[pks]>0] = 1
    direction[v_raw[pks]<=0] = -1
    
def detectLicks(licks,distance = 90,width = (10,60),height = 1):
    licks = sp.stats.zscore(licks)
    pks,_ = sp.signal.find_peaks(licks,height = height,distance = distance,width = width)
    return pks
           
def getLickRate(licks,lickTimes, w= 100):
    dt = np.median(np.diff(lickTimes[:,0]))*1000 # in millisecond
    w = int(np.round(w/dt))
    lcks = detectLicks(licks)  
    lickEvents = np.zeros(lickTimes.shape)
    lickEvents[lcks] = 1   
    lickEvents = pd.DataFrame(lickEvents)
    windows = lickEvents.rolling(w,min_periods=1)
    return (np.array(windows.sum())/0.1)

def cleanLickBursts(lcks,minTime=0.2):
    lastLickInd = len(lcks)
    li = 0
    removeLcks = []
    while li<lastLickInd-1:
        lck = lcks[li]
        lckDist = lcks[li+1]-lck
        if (lckDist<minTime):
            removeLcks.append(li+1)
        li+=1
        # else:
        #     lcks = np.delete(lcks,removeLcks)
        #     lastLickInd = len(lcks)
        #     removeLcks = []  
        #     li = 0
    lcks  = np.delete(lcks,removeLcks)
    return lcks