from Scripts.dataPrep import baseTrainingData, loadAdj
from Scripts.expSetUp import getTestTrainingData,appendPredictedCostToFeatures
from Scripts.methods import simpleSampling,OLSRegression,MLPRegression,GNNSimple
from Scripts.evaluation import performanceMetrics, writeResults
import csv
import os
from time import time
print('--Modules Imported--')

# Create results csv

runRef = str(int(time()))

resultsFileName = 'results/results_'+runRef+'.csv'

header = ['expNum','method','poi','stratum','probe','sampleRate','seedSplit','spatialCnx','absError','absErrorPcnt','jainsError','jainsPred','jainsError','correlation','corrConfidence','inferenceTime']

with open(resultsFileName,'a', newline='') as f:
    read_file = csv.writer(f)
    read_file.writerow(header)

# Create folder for plots

if not os.path.exists('Results/Plots/'+runRef):
    os.makedirs('Results/Plots/'+runRef)

# Model Parameters

device = "cpu"

features = ['avgEuclidDist','Shape__Are','distToUrbanCentre','populationDensity']
target = 'avgAccessCost'

#MLP
hiddenMLP = 64
epochsMLP = 500

#GNN
hidden1GNN = 64
hidden2GNN = 64
epochsGNN = 100

#Write as text file

fileName = 'results/settings'+runRef+'.txt'

with open(fileName, 'w') as f:
    f.write('Device : ' + str(device))
    f.write('\n')
    f.write('Features : ' + str(features))
    f.write('\n')
    f.write('Target : ' + str(target))
    f.write('\n')
    f.write('MLP Hidden Nodes : ' + str(hiddenMLP))
    f.write('\n')
    f.write('MLP Epochs : ' + str(epochsMLP))
    f.write('\n')
    f.write('GNN Hidden Layer 1 : ' + str(hidden1GNN))
    f.write('\n')
    f.write('GNN Hidden Layer 2 : ' + str(hidden2GNN))
    f.write('\n')
    f.write('GNN Epochs : ' + str(epochsGNN))
    f.write('\n')    

# Experiment Parameter sets

#pois =     ['Hospital', 'Job Centre', 'Strategic Centre', 'School', 'Childcare', 'Rail Station']
#stratums = ['Weekday (AM peak)','Weekday (PM peak)','Weekday (inter-peak)','Weekday (evening)','Saturday','Sunday']    
#probes =[0.9,0.7,0.5,0.4,0.3,0.2,0.15,0.1,0.05,0.04,0.03,0.02,0.01]
#sampleRates = [1,0.75,0.5,0.25,0.1]
#seedSplits = [0.9,0.7,0.5,0.3,0.1]
#spatialConnections = ['1hop-g', '1hop-mm', 'walk-g', 'walk-mm', 'euc-g', 'euc-mm']

pois =     ['Hospital', 'Job Centre', 'Strategic Centre', 'School', 'Childcare', 'Rail Station']
stratums = ['Weekday (AM peak)','Sunday']    
probes =[0.7,0.5,0.3,0.2,0.1,0.05,0.03,0.01]
sampleRates = [1,0.5,0.25]
seedSplits = [0.7,0.5,0.3]
spatialConnections = ['1hop-g', 'walk-g', 'euc-g']

# Static Parameters

dbloc = 'Data/tfwm.db'
shpLoc = 'Data/west_midlands_OAs/west_midlands_OAs.shp'
oaInfoLoc = 'Data/oa_info.csv'
adjLocDict = {
    '1hop-G' : 'Data/adjMx/adjMx_1hop_Gaus.npy',
    '1hop-MM' : 'Data/adjMx/adjMx_1hop_Gaus.npy',
    'walk-G' : 'Data/adjMx/adjMx_Walk_Gaus.npy',
    'walk-MM' : 'Data/adjMx/adjMx_Walk_MM.npy',
    'euc-G' : 'Data/adjMx/adjMx_euclid_Gaus.npy',
    'euc-MM' : 'Data/adjMx/adjMx_euclid_MM.npy',
    }

#%% Count experiments

countExp = 0

for p in pois:
    for s in stratums:
        for pb in probes:
            for sr in sampleRates:
                for ss in seedSplits:
                    countExp += 3
                    for aType in spatialConnections:
                        countExp += 2
print('Number of experiments : ' + str(countExp))

expNum = 0

#Run Experiments

for p in pois:
    for s in stratums:
        baseData, oaIndex = baseTrainingData(dbloc,shpLoc,oaInfoLoc,s,p)
        for pb in probes:
            for sr in sampleRates:
                runOnce = False
                for ss in seedSplits:
                    x, y, ySample, testMask, trainMask, seedMask, seedTrainMask, baseData, numSPQ = getTestTrainingData(p,s,baseData,oaIndex,dbloc,pb,sr,ss,features,target='avgAccessCost')
                    while runOnce == False:
                        #Sampling
                        expNum += 1
                        print()
                        print(expNum)
                        print(expNum/countExp)
                        predSample, infTime = simpleSampling(baseData, testMask)
                        method = 'Sampling'
                        aType = 'None'
                        absError,absErrorPcnt,jainActual,jainPred,jainsError,correation,corrConfidence = performanceMetrics(predSample,baseData,testMask,y,shpLoc,oaInfoLoc,runRef,expNum)
                        writeResults(expNum,method,p, s, pb, sr, ss, aType, absError,absErrorPcnt,jainActual,jainPred,jainsError,correation,corrConfidence,infTime,resultsFileName)
                        
                        #OLS Regression
                        expNum += 1
                        print()
                        print(expNum)
                        print(expNum/countExp)
                        predOLS, infTime = OLSRegression(x,y,trainMask,testMask)
                        method = 'Regr-OLS'
                        aType = 'None'
                        absError,absErrorPcnt,jainActual,jainPred,jainsError,correation,corrConfidence = performanceMetrics(predOLS,baseData,testMask,y,shpLoc,oaInfoLoc,runRef,expNum)
                        writeResults(expNum,method,p, s, pb, sr, ss, aType, absError,absErrorPcnt,jainActual,jainPred,jainsError,correation,corrConfidence,infTime,resultsFileName)
                        
                        #MLP Regression
                        expNum += 1
                        print()
                        print(expNum)
                        print(expNum/countExp)
                        predMLP, infTime = MLPRegression(x,y,trainMask,testMask,hiddenMLP,epochsMLP, device)
                        method = 'Regr-MLP'
                        aType = 'None'
                        absError,absErrorPcnt,jainActual,jainPred,jainsError,correation,corrConfidence = performanceMetrics(predMLP,baseData,testMask,y,shpLoc,oaInfoLoc,runRef,expNum)
                        writeResults(expNum,method,p, s, pb, sr, ss, aType, absError,absErrorPcnt,jainActual,jainPred,jainsError,correation,corrConfidence,infTime,resultsFileName)
                        runOnce == True
                    
                    for aType in spatialConnections:
                        aType = '1hop-G'
                        adjLoc = adjLocDict[aType]
                        edgeIndexNp,edgeWeightsNp = loadAdj(adjLoc,oaIndex)
                        
                        #GNN Simple
                        expNum += 1
                        print()
                        print(expNum)
                        print(expNum/countExp)
                        predGNNSimple, infTime = GNNSimple(x,ySample,device,edgeIndexNp,edgeWeightsNp,hidden1GNN,hidden2GNN,epochsGNN,trainMask,testMask)
                        method = 'GNN-Simple'
                        absError,absErrorPcnt,jainActual,jainPred,jainsError,correation,corrConfidence = performanceMetrics(predGNNSimple,baseData,testMask,y,shpLoc,oaInfoLoc,runRef,expNum)
                        writeResults(expNum,method,p, s, pb, sr, ss, aType, absError,absErrorPcnt,jainActual,jainPred,jainsError,correation,corrConfidence,infTime,resultsFileName)

                        # GNN Seeds
                        expNum += 1
                        print()
                        print(expNum)
                        print(expNum/countExp)
                        _x = appendPredictedCostToFeatures(baseData,seedMask,features,x,target='sampleAccessCost')
                        predGNNSeeds, infTime = GNNSimple(_x,ySample,device,edgeIndexNp,edgeWeightsNp,hidden1GNN,hidden2GNN,epochsGNN,seedTrainMask,testMask)
                        method = 'GNN-Seeds'
                        absError,absErrorPcnt,jainActual,jainPred,jainsError,correation,corrConfidence = performanceMetrics(predGNNSeeds,baseData,testMask,y,shpLoc,oaInfoLoc,runRef,expNum)
                        writeResults(expNum,method,p, s, pb, sr, ss, aType, absError,absErrorPcnt,jainActual,jainPred,jainsError,correation,corrConfidence,infTime,resultsFileName)

