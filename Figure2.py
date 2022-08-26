# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 11:04:54 2022

@author: LABadmin
"""
# Load Data
tups_ = np.vstack((sessId,nId)).T
tups_ = tups_[r[0,:]==1,:]
modelsBase = GetOneFunctionVarianceDifferentModels(dataDir = 'D:\\Figures\\OneFunction_raw\\',cutoff=0.00001,shuffles= True,resp=tups_)
nullProps1 = np.load('D:\\Figures\\Paper_Feedback\\Fitting\\Cumulative\\nulProps1.npy')
nullProps2 = np.load('D:\\Figures\\Paper_Feedback\\Fitting\\Cumulative\\nulProps2.npy')
nullProps3 = np.load('D:\\Figures\\Paper_Feedback\\Fitting\\Cumulative\\nulProps3.npy')
nullProps4 = np.load('D:\\Figures\\Paper_Feedback\\Fitting\\Cumulative\\nulProps4.npy')
#%% Panel A
PrintResponseByAllGroups(AllData, iteration = [16],nn=[107],saveDir = 'D:\\Temp\\')
#%% Panel B
fits = np.array(modelsBase['fit'][modelsBase['Id']==(16,107)])
GetFitFromFitList(fits,'P')
#%% Panel C
PrintResponseByAllGroups(AllData, iteration = [16],nn=[65],saveDir = 'D:\\Temp\\')

#%% Panel D
fits = np.array(modelsBase['fit'][modelsBase['Id']==(16,65)])
GetFitFromFitList(fits,'P')
#%% Panel E
PrintResponseByAllGroups(AllData, iteration = [16],nn=[136],saveDir = 'D:\\Temp\\')

#%% Panel F
fits = np.array(modelsBase['fit'][modelsBase['Id']==(16,136)])
GetFitFromFitList(fits,'F')

#%% Paenls G-N
def CreateNullIntervalPlot(ax,dist):
    ys = []
    xs = []
    f,axt = plt.subplots(1)
    # sns.kdeplot(data=dist.T,legend=False,common_norm=False,cumulative=True,ax=axt,clip=(-2,2))
    sns.kdeplot(data=dist,ax=axt,legend=False,common_norm=False,common_grid=True, cumulative=True,clip=(-2,2))

    for i in range(0,len(axt.lines)):     
        line = axt.lines[i]
        x, y = line.get_data()
        y/=np.max(y)
        ys.append(y)
        xs.append(x)
        plt.close(f)
    ys = np.vstack(ys)
    xs = np.vstack(xs)
       
    ax.fill_between(x,np.nanpercentile(ys, 2.5,axis=0),np.nanpercentile(ys, 97.5,axis=0),color = 'grey',alpha=0.4)
    
    return None

#Find the R for pupil and feedback
f1 = np.array(modelsBase[modelsBase['bestFit']=='F']['fit'])
fp1 = np.array(modelsBase[modelsBase['bestFit']=='PF']['fit'])
p1 = np.array(modelsBase[modelsBase['bestFit']=='P']['fit'])
nf1 = np.array(modelsBase[modelsBase['bestFit']!='F']['fit'])
np1 = np.array(modelsBase[modelsBase['bestFit']!='P']['fit'])
nfp1 = np.array(modelsBase[modelsBase['bestFit']!='PF']['fit'])
g1 = np.array(modelsBase[modelsBase['bestFit']=='G']['fit'])
c1 = np.array(modelsBase[modelsBase['bestFit']=='C']['fit'])
ng1 = np.array(modelsBase[modelsBase['bestFit']!='G']['fit'])
nc1 = np.array(modelsBase[modelsBase['bestFit']!='C']['fit'])
gc1 = np.array(modelsBase[modelsBase['bestFit']!='GC']['fit'])
allFit = np.array(modelsBase['fit'])


fr= GetRFromFitList(list(f1),'F')
fpr= GetRFromFitList(list(fp1),'PF')
nfr = GetRFromFitList(list(nf1),'F')
pr = GetRFromFitList(list(p1),'P')
npr = GetRFromFitList(list(np1),'P')
nfpr = GetRFromFitList(list(nfp1),'PF')
gr = GetRFromFitList(list(g1),'G')
ngr = GetRFromFitList(list(ng1),'G')
cr = GetRFromFitList(list(c1),'C')
ncr = GetRFromFitList(list(nc1),'C')

frall= GetRFromFitList(list(allFit),'F')
prall = GetRFromFitList(list(allFit),'P')
grall = GetRFromFitList(list(allFit),'G')
crall = GetRFromFitList(list(allFit),'C')


##### Feedback Plotting #########
f,ax = plt.subplots(1)
ax.scatter(nfr[:,0]+nfr[:,1],nfr[:,0],s=30,c=['w'],edgecolors='grey')
ax.scatter(fr[:,0]+fr[:,1],fr[:,0],s=30,c=['k'],edgecolors='k')
ax.scatter(fpr[:,0]+fpr[:,2],fpr[:,0],s=30,c=['k'],edgecolors='k')
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.plot(np.arange(0,0.7,0.1),np.arange(0,0.7,0.1),'k--')
ax.set_title('R (gain) feedback')
ax.set_xlabel('Positive Feedback')
ax.set_ylabel('Negative Feedback')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
f.set_size_inches(8, 8)

# Extra histogram to put diagonally
f,ax = plt.subplots(1)

valsPos_ns = nullProps1[:,:,[0,4]]
valsPos_ns = valsPos_ns[:,:,1]/(0.5*(2*valsPos_ns[:,:,0]+valsPos_ns[:,:,1]))
valsPos = (frall[:,0]+frall[:,1]-frall[:,0])/(0.5*(frall[:,0]+frall[:,1]+frall[:,0]))

sns.kdeplot(data=[valsPos],palette=['black'],ax=ax,legend=False,common_norm=False,cumulative=True,clip=(-2,2))
sorted_data = np.sort(valsPos)
y = np.arange(sorted_data.size)
y=y/np.max(y)
CreateNullIntervalPlot(ax,valsPos_ns)
plt.grid(False)
ax.set_xlabel('Respone Modulation (%)')
ax.set_ylabel('Proportion of Neurons')
ax.set_xlim(-2,2)
ax.set_ylim(0,1)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
f.set_size_inches(np.sqrt(8), np.sqrt(8))
ax.vlines(0,0,1,'k')
##### Pupil Plotting #########
f,ax = plt.subplots(1)
ax.scatter(npr[:,0]+npr[:,1],npr[:,0],s=30,c=['w'],edgecolors='grey')
ax.scatter(pr[:,0]+pr[:,1],pr[:,0],s=30,c=['k'],edgecolors='k')
ax.scatter(fpr[:,0]+fpr[:,1],fpr[:,0],s=30,c=['k'],edgecolors='k')
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.plot(np.arange(0,0.7,0.1),np.arange(0,0.7,0.1),'k--')
ax.set_title('R (gain) Pupil')
ax.set_xlabel('Large Pupil')
ax.set_ylabel('Small Pupil')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
f.set_size_inches(8, 8)

# Extra histogram to put diagonally
f,ax = plt.subplots(1)

valsPos_ns = nullProps2[:,:,[0,1]]
valsPos_ns = valsPos_ns[:,:,1]/(0.5*(2*valsPos_ns[:,:,0]+valsPos_ns[:,:,1]))
valsPos = (prall[:,1])/(0.5*(2*prall[:,0]+prall[:,1]))
sns.kdeplot(data=[valsPos],palette=['black'],ax=ax,legend=False,common_norm=False,cumulative=True,clip=(-2,2))

CreateNullIntervalPlot(ax,valsPos_ns)
plt.grid(False)
ax.set_xlabel('Respone Modulation (%)')
ax.set_ylabel('Proportion of Neurons')
ax.set_xlim(-2,2)
ax.set_ylim(0,1)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
f.set_size_inches(np.sqrt(8), np.sqrt(8))
ax.vlines(0,0,1,'k')

##### Pupil Feedback Plotting #########
f,ax = plt.subplots(1)
###OPTIONAL

a = (nfpr[:,1])/(0.5*(2*nfpr[:,0]+nfpr[:,1]))
b = (nfpr[:,2])/(0.5*(2*nfpr[:,0]+nfpr[:,2]))
ax.scatter(a,b,s=30,c=(0.6,0.6,0.6))
a = (fpr[:,1])/(0.5*(2*fpr[:,0]+fpr[:,1]))
b = (fpr[:,2])/(0.5*(2*fpr[:,0]+fpr[:,2]))

ax.scatter(a,b,s=30,c=['k'])


ax.set_aspect('equal', adjustable='box')

ax.hlines(0,-2,2,'k',linestyles='dashed')
ax.vlines(0,-2,2,'k',linestyles='dashed')
ax.set_title('Pupil gain vs. Feeback gain')
ax.set_xlabel('Pupil gain')
ax.set_ylabel('Feedback gain')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)


f,ax = plt.subplots(1)
ax.scatter(np.abs(fpr[:,1]),np.abs(fpr[:,2]),s=30,c=['k'],edgecolors='k')
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(0,0.45)
ax.set_ylim(0,0.45)
ax.plot(np.arange(-0.7,0.7,0.1),np.arange(-0.7,0.7,0.1),'k--')
ax.set_title('Pupil gain vs. Feeback gain')
ax.set_xlabel('Pupil gain')
ax.set_ylabel('Feedback gain')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

##### GoNoGo Plotting #########
f,ax = plt.subplots(1)
ax.scatter(ngr[:,0]+ngr[:,1],ngr[:,0],s=30,c=['w'],edgecolors='grey')
ax.scatter(gr[:,0]+gr[:,1],gr[:,0],s=30,c=['k'],edgecolors='k')
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.plot(np.arange(0,0.7,0.1),np.arange(0,0.7,0.1),'k--')
ax.set_title('R (gain) Go')
ax.set_xlabel('Go')
ax.set_ylabel('No Go')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
f.set_size_inches(8, 8)

# Extra histogram to put diagonally
f,ax = plt.subplots(1)

valsPos_ns = nullProps3[:,:,[0,2]]
valsPos_ns = valsPos_ns[:,:,1]/(0.5*(2*valsPos_ns[:,:,0]+valsPos_ns[:,:,1]))
valsPos = (grall[:,1])/(0.5*(2*grall[:,0]+grall[:,1]))
sns.kdeplot(data=[valsPos],palette=['black'],ax=ax,legend=False,common_norm=False,cumulative=True,clip=(-2,2))

CreateNullIntervalPlot(ax,valsPos_ns)
plt.grid(False)
ax.set_xlabel('Respone Modulation (%)')
ax.set_ylabel('Proportion of Neurons')
ax.set_xlim(-2,2)
ax.set_ylim(0,1)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
f.set_size_inches(np.sqrt(8), np.sqrt(8))
ax.vlines(0,0,1,'k')
##### Correctness Plotting #########
f,ax = plt.subplots(1)
ax.scatter(ncr[:,0]+ncr[:,1],ncr[:,0],s=30,c=['w'],edgecolors='grey')
ax.scatter(cr[:,0]+cr[:,1],cr[:,0],s=30,c=['k'],edgecolors='k')
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.plot(np.arange(0,0.7,0.1),np.arange(0,0.7,0.1),'k--')
ax.set_title('R (gain) Correct')
ax.set_xlabel('Correct')
ax.set_ylabel('Incorrect')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
f.set_size_inches(8, 8)

# Extra histogram to put diagonally
f,ax = plt.subplots(1)

valsPos_ns = nullProps4[:,:,[0,3]]
valsPos_ns = valsPos_ns[:,:,1]/(0.5*(2*valsPos_ns[:,:,0]+valsPos_ns[:,:,1]))
valsPos = (crall[:,1])/(0.5*(2*crall[:,0]+crall[:,1]))
sns.kdeplot(data=[valsPos],palette=['black'],ax=ax,legend=False,common_norm=False,cumulative=True,clip=(-2,2))

CreateNullIntervalPlot(ax,valsPos_ns)
plt.grid(False)
ax.set_xlabel('Respone Modulation (%)')
ax.set_ylabel('Proportion of Neurons')
ax.set_xlim(-2,2)
ax.set_ylim(0,1)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
f.set_size_inches(np.sqrt(8), np.sqrt(8))
ax.vlines(0,0,1,'k')

#%% Panels O-R
dataDir = 'D:\\Figures\\OneFunction_raw\\'

plt.close('all')


reps = 10

predictionImprov = np.zeros((5,16))*np.nan
dfs = []
i = 0
ms = []
control = None

shuffleNumber = 100
shuffleDist = np.zeros((4,shuffleNumber))

ress = []

#States = 2: pupil, 3: Action, 4: outcome, 5: reward

for p in [2,3,4,5]:
    results,m = PredictionPerformanceSpecificPopulation(testStat=p,reps=reps,split=0.5,controlStat=control,C=5,controlState=0)   
    
    ms.append(m)  
    f,ax = plt.subplots(1)    
      
    ax.scatter(np.nanmean(results,-1)[:,0],np.nanmean(results,-1)[:,1],c='k',edgecolors='black')
   
    ax.plot(np.arange(-1,1,0.1),np.arange(-1,1,0.1),'k--')
    ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')  
    f.set_size_inches(8, 8)
    ax.set_ylim(-0.2,1)    
    ax.set_xlim(-0.2,1)  
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # res = results[:,1,:]-results[:,0,:]
    res = np.nanmean(results,-1)
    res = res[:,1]-res[:,0]
    predictionImprov[i,:] = res
    i+=1


