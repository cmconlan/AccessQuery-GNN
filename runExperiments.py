from Scripts.dataPrep import getBaseTrainingData, loadAdj
from Scripts.expSetUp import getTestTrainingData,appendPredictedCostToFeatures,constructAdjMx
from Scripts.methods import simpleSampling,OLSRegression,MLPRegression,GNNSimple
from Scripts.evaluation import getPerformanceMetrics, writeResults
from Scripts.util.database import Database
import csv
import os
from time import time
import yaml
import sys

print('--Modules Imported--')

# Import experiment yaml

#ymlFile = 'exp1'
ymlFile = sys.argv[1]
print('YAML File : ' + str(ymlFile))

with open('Experiments/'+ymlFile + '.yml', 'r') as stream:
    experimentParams = yaml.safe_load(stream)

# Set up results environment

resultsFileName = 'Results/results_'+ymlFile+'.csv'

try:
    os.remove(resultsFileName)
except OSError:
    pass

header = ['expNum','method','poi','stratum','budget','sampleRate','seedSplit','AL','absError','absErrorPcnt','jainsActual','jainsPred','jainsError','correlation','corrConfidence','inferenceTime','numSPQ']

with open(resultsFileName,'a', newline='') as f:
    read_file = csv.writer(f)
    read_file.writerow(header)

# Create folder for plots

if not os.path.exists('Results/Plots/'+ymlFile):
    os.makedirs('Results/Plots/'+ymlFile)
    
if not os.path.exists('Results/Data/'+ymlFile):
    os.makedirs('Results/Data/'+ymlFile)

# Environment variables

shpFileLoc = 'Data/west_midlands_OAs/west_midlands_OAs.shp'
oaInfoLoc = 'Data/oa_info.csv'
dbLoc= 'Data/access.db'
euclidPath = 'Data/bus_network/euclidMatrix.csv'

# Experiment variables
#model features
mf = experimentParams['modelFeatures']
mfAcc = experimentParams['predictFeatures']
k = experimentParams['k']
m = experimentParams['adjMxType']

#Model variables
hiddenMLP = experimentParams['MLPParameters']['hiddenMLP']
epochsMLP = experimentParams['MLPParameters']['epochs']

hidden1GNN = experimentParams['GNNParameters']['hidden1GNN']
hidden2GNN = experimentParams['GNNParameters']['hidden2GNN']
epochsGNN = experimentParams['GNNParameters']['epochs']
device = experimentParams['device']

#%%
expNum = 0
#Use placehold value in first instance
ss = 0.5
for p in experimentParams['POIsToTest']:
    for s in experimentParams['stratemsToTest']:
        #Get base training data (e.g., OAs with relevant features from POI and stratum)
        baseData, oaIndex, poiLonLat, poiInd = getBaseTrainingData(shpFileLoc,oaInfoLoc,s,p,dbLoc)
        
        for pb in experimentParams['budgetsToTest']:
            for sr in experimentParams['sampleRatesToTest']:
                for al in experimentParams['ALToTest']:
                    #Construct training matrices
                    x, y, ySample, testMask, trainMask, seedMask, seedTrainMask, baseData, numSPQ, numFullSample, scalerY, scalerYSample = getTestTrainingData(baseData,pb,al,oaIndex,ss,dbLoc,poiInd,s,sr,mf)
                    
                    #Construct matrix
                    featureForClustering = baseData[mf].to_numpy()
                    adjMx = constructAdjMx(k,euclidPath,m,featureForClustering,oaIndex)
                    edgeIndexNp,edgeWeightsNp = loadAdj(adjMx,oaIndex)
                    
                    if experimentParams['modelsToRun']['OLS']:
                        #OLS Regression
                        expNum += 1
                        print('Experiment : ' + str(expNum))
                        method = 'Regr-OLS'
                        predVector, infTime = OLSRegression(x,y,trainMask,testMask)
                        absError,absErrorPcnt,jainActual,jainPred,jainsError,correation,corrConfidence,baseData = getPerformanceMetrics(testMask,scalerY,predVector,baseData,y,shpFileLoc,trainMask,poiLonLat,ymlFile,expNum)
                        writeResults(expNum,method,p, s, pb, sr, ss, al, absError,absErrorPcnt,jainActual,jainPred,jainsError,correation,corrConfidence,infTime,numSPQ,resultsFileName,baseData,ymlFile)
                    if experimentParams['modelsToRun']['MLP']:
                        expNum += 1
                        print('Experiment : ' + str(expNum))
                        method = 'Regr-MLP'
                        proceedMLP = False
                        while proceedMLP == False:
                            predVector, infTime, losses = MLPRegression(x,y,trainMask,testMask,hiddenMLP,epochsMLP, device)                                
                            if float(losses[-1].cpu().detach().numpy()) / float(losses[0].cpu().detach().numpy()) < 0.95:
                                proceedMLP = True
                        print()
                        print('Evaluating MLP')
                        absError,absErrorPcnt,jainActual,jainPred,jainsError,correation,corrConfidence,baseData = getPerformanceMetrics(testMask,scalerY,predVector,baseData,y,shpFileLoc,trainMask,poiLonLat,ymlFile,expNum)
                        writeResults(expNum,method,p, s, pb, sr, ss, al, absError,absErrorPcnt,jainActual,jainPred,jainsError,correation,corrConfidence,infTime,numSPQ,resultsFileName,baseData,ymlFile)
                    
                    if experimentParams['modelsToRun']['GNNSimple']:
                        #GNN Simple
                        expNum += 1
                        print('Experiment : ' + str(expNum))
                        method = 'GNN-Simple'
                        proceedGNNSimpl = False
                        while proceedGNNSimpl == False:
                            predVector, infTime, losses = GNNSimple(x,ySample,device,edgeIndexNp,edgeWeightsNp,hidden1GNN,hidden2GNN,epochsGNN,trainMask,testMask)
                            if float(losses[-1].cpu().detach().numpy()) / float(losses[0].cpu().detach().numpy()) < 0.95:
                                proceedMLP = True
                        print()
                        print('Evaluating GNN Simple')
                        absError,absErrorPcnt,jainActual,jainPred,jainsError,correation,corrConfidence,baseData = getPerformanceMetrics(testMask,scalerY,predVector,baseData,y,shpFileLoc,trainMask,poiLonLat,ymlFile,expNum)
                        writeResults(expNum,method,p, s, pb, sr, ss, al, absError,absErrorPcnt,jainActual,jainPred,jainsError,correation,corrConfidence,infTime,numSPQ,resultsFileName,baseData,ymlFile)
                    if experimentParams['modelsToRun']['GNNSeeds']:
                        for ss in experimentParams['seedSplitsToTest']:
                            #Construct training matrices
                            x, y, ySample, testMask, trainMask, seedMask, seedTrainMask, baseData, numSPQ, numFullSample, scalerY, scalerYSample = getTestTrainingData(baseData,pb,al,oaIndex,ss,dbLoc,poiInd,s,sr,mf)
                            
                            #Construct matrix
                            featureForClustering = baseData[mf].to_numpy()
                            adjMx = constructAdjMx(k,euclidPath,m,featureForClustering,oaIndex)
                            edgeIndexNp,edgeWeightsNp = loadAdj(adjMx,oaIndex)
                            
                            #GNN with Seeds
                            expNum += 1
                            print('Experiment : ' + str(expNum))
                            method = 'GNN-Seeds'
                            _x = appendPredictedCostToFeatures(baseData,seedMask,mfAcc,x,target='sampleAccessCost')
                            proceedGNNSeeds = False
                            while proceedGNNSeeds == False:
                                predVector, infTime, losses = GNNSimple(_x,ySample,device,edgeIndexNp,edgeWeightsNp,hidden1GNN,hidden2GNN,epochsGNN,seedTrainMask,testMask)
                                if float(losses[-1].cpu().detach().numpy()) / float(losses[0].cpu().detach().numpy()) < 0.95:
                                    proceedGNNSeeds = True
                            print()
                            print('Evaluating GNN Seeds')
                            absError,absErrorPcnt,jainActual,jainPred,jainsError,correation,corrConfidence,baseData = getPerformanceMetrics(testMask,scalerY,predVector,baseData,y,shpFileLoc,trainMask,poiLonLat,ymlFile,expNum)
                            writeResults(expNum,method,p, s, pb, sr, ss, al, absError,absErrorPcnt,jainActual,jainPred,jainsError,correation,corrConfidence,infTime,numSPQ,resultsFileName,baseData,ymlFile)
                        

#%%
# expNum = 0

# #%%
# p = experimentParams['POIsToTest'][2]
# s = experimentParams['stratemsToTest'][0]

# baseData, oaIndex, poiLonLat, poiInd = getBaseTrainingData(shpFileLoc,oaInfoLoc,s,p,dbLoc)

# #%%

# pb = experimentParams['budgetsToTest'][1]
# sr = experimentParams['sampleRatesToTest'][0]
# ss = experimentParams['seedSplitsToTest'][0]
# al = experimentParams['ALToTest'][0]

# #Construct training matrices
# x, y, ySample, testMask, trainMask, seedMask, seedTrainMask, baseData, numSPQ, numFullSample, scalerY, scalerYSample = getTestTrainingData(baseData,pb,al,oaIndex,ss,dbLoc,poiInd,s,sr,mf)

# #Construct matrix
# featureForClustering = baseData[mf].to_numpy()
# adjMx = constructAdjMx(k,euclidPath,m,featureForClustering,oaIndex)
# edgeIndexNp,edgeWeightsNp = loadAdj(adjMx,oaIndex)

# #%%
# #OLS Regression
# expNum += 1
# method = 'Regr-OLS'
# predVector, infTime = OLSRegression(x,y,trainMask,testMask)
# absError,absErrorPcnt,jainActual,jainPred,jainsError,correation,corrConfidence,baseData = getPerformanceMetrics(testMask,scalerY,predVector,baseData,y,shpFileLoc,trainMask,poiLonLat,ymlFile,expNum)
# writeResults(expNum,method,p, s, pb, sr, ss, al, absError,absErrorPcnt,jainActual,jainPred,jainsError,correation,corrConfidence,infTime,numSPQ,resultsFileName,baseData,ymlFile)

# #%%
# #MLP Regression

# predVector, infTime, losses = MLPRegression(x,y,trainMask,testMask,hiddenMLP,epochsMLP, device)
# print(predVector.sum())
# print(float(losses[-1].cpu().detach().numpy()) / float(losses[0].cpu().detach().numpy()))

# #%%



# #%%
# expNum += 1
# method = 'Regr-MLP'

# proceed = False
# while proceed == False:
#     predVector, infTime, losses = MLPRegression(x,y,trainMask,testMask,hiddenMLP,epochsMLP, device)

#     print(predVector.sum())
    
#     if predVector.sum() > 10:
#         proceed = True

# #%%
# absError,absErrorPcnt,jainActual,jainPred,jainsError,correation,corrConfidence,baseData = getPerformanceMetrics(testMask,scalerY,predVector,baseData,y,shpFileLoc,trainMask,poiLonLat,ymlFile,expNum)
# #%%
# writeResults(expNum,method,p, s, pb, sr, ss, al, absError,absErrorPcnt,jainActual,jainPred,jainsError,correation,corrConfidence,infTime,numSPQ,resultsFileName,baseData,ymlFile)

# #%%
# #GNN Simple
# expNum += 1
# method = 'GNN-Simple'
# predVector, infTime = GNNSimple(x,ySample,device,edgeIndexNp,edgeWeightsNp,hidden1GNN,hidden2GNN,epochsGNN,trainMask,testMask)
# absError,absErrorPcnt,jainActual,jainPred,jainsError,correation,corrConfidence,baseData = getPerformanceMetrics(testMask,scalerY,predVector,baseData,y,shpFileLoc,trainMask,poiLonLat,ymlFile,expNum)
# writeResults(expNum,method,p, s, pb, sr, ss, al, absError,absErrorPcnt,jainActual,jainPred,jainsError,correation,corrConfidence,infTime,numSPQ,resultsFileName,baseData,ymlFile)

# #%%
# #GNN with Seeds
# expNum += 1
# method = 'GNN-Seeds'
# _x = appendPredictedCostToFeatures(baseData,seedMask,mfAcc,x,target='sampleAccessCost')
# predVector, infTime = GNNSimple(_x,ySample,device,edgeIndexNp,edgeWeightsNp,hidden1GNN,hidden2GNN,epochsGNN,seedTrainMask,testMask)
# absError,absErrorPcnt,jainActual,jainPred,jainsError,correation,corrConfidence,baseData = getPerformanceMetrics(testMask,scalerY,predVector,baseData,y,shpFileLoc,trainMask,poiLonLat,ymlFile,expNum)
# writeResults(expNum,method,p, s, pb, sr, ss, al, absError,absErrorPcnt,jainActual,jainPred,jainsError,correation,corrConfidence,infTime,numSPQ,resultsFileName,baseData,ymlFile)