# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 11:31:29 2022

@author: LABadmin
"""

def GetOneFunctionVarianceDifferentModels(dataDir = 'D:\\Figures\\OneFunction_raw\\',flat = False, diffCutoff=0.01, cutoff = 0.01,return_subset = False,depth = None,shuffles = False,resp = []):
    # See what version of the function is the best to determine which behavioural factor affects it the most
    #pgcf
    subsets = [[1],[2],[3],[4],[1,2],[1,3],[1,4],[2,3],[2,4],[3,4],[1,2,3],[1,2,4],[1,3,4],[2,3,4]]
    subsetText = ['None','P', 'G','C','F','PG','PC','PF','GC','GF','CF','PGC','PGF','PCF','GCF','PGCF']
    subsetCount = np.zeros(16)
    modelFitList = []
    for fname in glob.glob(dataDir+ 'Metadata*.pickle', recursive=True):  
        sessionNumbers = np.array(re.findall('\d+',fname)).astype(int) 
        
        # check that neuron is responsive
        if (len(resp)>0):
            RespNeurons = resp[:,1:][resp[:,0]==sessionNumbers[0]]
            if len(sessionNumbers)>2:
                RespNeurons = RespNeurons[:,-1][RespNeurons[:,0]==sessionNumbers[1]]
            if not(sessionNumbers[-1] in RespNeurons):
                continue
                
        if (shuffles):
            shuffleFileName = 'Data'
            if len(sessionNumbers)>2:
                shuffleFileName += str(sessionNumbers[0]) + '.' + str(sessionNumbers[1])
            else:
                shuffleFileName += str(sessionNumbers[0])
            shuffleFileName+='_shuffle.pickle'   
            
                
            with open(dataDir+shuffleFileName, 'rb') as f: 
               shuffleRes = pickle.load(f)   
            mainShuff = shuffleRes[0]
            catShuff = shuffleRes[1]
            cutoff = np.nanpercentile(mainShuff[:,sessionNumbers[-1]],97.5)
            catShuff = catShuff[:,:,[sessionNumbers[-1]]]
               
        if (not (depth is None)):           
            with open(dataDir+'Data'+str(sessionNumbers[0])+'.'+str(sessionNumbers[1])+'.pickle', 'rb') as f:
                generalRes = pickle.load(f)        
            neuronDepth = generalRes['scDepth'] - generalRes['depths'][int(sessionNumbers[2])]
            if not ((neuronDepth>depth[0]) & (neuronDepth<depth[1])):
                continue;
        with open(fname, 'rb') as f:
            res = pickle.load(f)[0]       
            if (flat):
                
                if (len(res['varFlatInclude'])==0):
                    continue
               
                if res['varFlatInclude'][:,0]>cutoff:#(res['varFlat'][0]>cutoff):
                   varsInclude = res['varFlatInclude'][:,0]
                   
                   # Shuffle Test the base
                   
                   
                   I = np.argmax(varsInclude)
                   maxVal = varsInclude[I]
                   addVar = maxVal-varsInclude[0]
                   
                   if (shuffles):
                       diffCutoff = np.nanpercentile(catShuff[I,:],95)
                   
                   if (maxVal>cutoff):
                       if (addVar<diffCutoff):
                            I=0
                       subsetCount[I]+=1
                       identifiers = tuple(np.array(re.findall('\d+',fname)).astype(int))
                       modelFitList.append({'Id':identifiers,'fit':res,'bestFit':subsetText[I]})
            else:
                if (len(res['varInclude'])==0):
                    continue
               
                varsInclude = res['varInclude'][:,0]
                if (len(res['varFlatInclude'])>0):
                    fitCutoff = res['varFlatInclude'][0,0]
                else:
                    fitCutoff = 0
                if (varsInclude[0]> fitCutoff):#(varsInclude[0]>cutoff):#(res['var'][0]>cutoff):
                   varsInclude = res['varInclude'][:,0]
                   I = np.argmax(varsInclude)
                   maxVal = varsInclude[I]
                   addVar = maxVal-varsInclude[0]  
                   
                   if (shuffles):
                        diffCutoff = np.nanpercentile(catShuff[I,:],95)
                   
                   if (maxVal>cutoff):
                       if (addVar<diffCutoff):
                            I=0
                       subsetCount[I]+=1
                       identifiers = tuple(np.array(re.findall('\d+',fname)).astype(int))
                       modelFitList.append({'Id':identifiers,'fit':res,'normal_var':varsInclude[0],'added_var':addVar,'bestFit':subsetText[I]})
    
    if (return_subset):
        return subsetText,subsetCount
    return pd.DataFrame(modelFitList)