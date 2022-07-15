from Scripts.dataPrep import getBaseTrainingData, loadAdj
from Scripts.expSetUp import getTestTrainingData,appendPredictedCostToFeatures,constructAdjMx
from Scripts.methods import simpleSampling,OLSRegression,MLPRegression,GNNSimple
from Scripts.evaluation import getPerformanceMetrics, writeResults
from Scripts.util.database import Database
import csv
import os
import time
import yaml
import sys
from Baselines.train_models import VATTrainer
from Baselines import coreg
import shutil
import numpy as np
import random

class CoregTrainer:
    """
    Coreg trainer class.
    """
    def __init__(self, data_dir, results_dir, num_train, num_trials,x,y,trainMask,testMask, k1=3,
                 k2=3, p1=2, p2=5, max_iters=100, pool_size=100,
                 batch_unlabeled=0, verbose=False):
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.num_train = num_train
        self.num_trials = num_trials
        
        self.X_labeled = x[trainMask]
        self.X_unlabeled = x[testMask]
        self.y_labeled = y[trainMask].reshape(-1, 1)
        self.y_unlabeled = y[testMask].reshape(-1, 1)

        self.k1 = k1
        self.k2 = k2
        self.p1 = p1
        self.p2 = p2
        self.max_iters = max_iters
        self.pool_size = pool_size
        self.batch_unlabeled = batch_unlabeled
        self.verbose = verbose
        #self._remake_dir(os.path.join(self.results_dir, 'results'))
        self._setup_coreg()

    def run_trials(self):
        """
        Run multiple trials of training.
        """
        self.coreg_model.run_trials(self.num_train, self.num_trials, self.verbose)
        self.test_hat = self.coreg_model.test_hat
        #self._write_results()

    def _remake_dir(self, directory):
        """
        If directory exists, delete it and remake it, if it doesn't exist, make
        it.
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
        else:
            shutil.rmtree(directory)
            os.makedirs(directory)

    def _setup_coreg(self):
        """
        Sets up coreg regressors and load data.
        """
        self.coreg_model = coreg.Coreg(
            self.X_labeled,self.X_unlabeled,self.y_labeled,self.y_unlabeled,self.k1, self.k2, self.p1, self.p2, self.max_iters, self.pool_size)
        #self.coreg_model.add_data(self.data_dir)

    def _write_results(self):
        """
        Writes results produced by experiment.
        """
        np.save(os.path.join(self.results_dir, 'results/mses1_train'),
                self.coreg_model.mses1_train)
        np.save(os.path.join(self.results_dir, 'results/mses1_test'),
                self.coreg_model.mses1_test)
        np.save(os.path.join(self.results_dir, 'results/mses2_train'),
                self.coreg_model.mses2_train)
        np.save(os.path.join(self.results_dir, 'results/mses2_test'),
                self.coreg_model.mses2_test)
        np.save(os.path.join(self.results_dir, 'results/mses_train'),
                self.coreg_model.mses_train)
        np.save(os.path.join(self.results_dir, 'results/mses_test'),
                self.coreg_model.mses_test)

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

#Baseline Param

data_dir = ''
results_dir = ''
num_train = 100
num_trials = 1
max_iters = 25

#%%
expNum = 0
experimentsTimeOut = []
#Use placehold value in first instance
ss = 0.5
for p in experimentParams['POIsToTest']:
    for s in experimentParams['stratemsToTest']:
        #Get base training data (e.g., OAs with relevant features from POI and stratum)
        baseData, oaIndex, poiLonLat, poiInd = getBaseTrainingData(shpFileLoc,oaInfoLoc,s,p,dbLoc)
        
        for pb in experimentParams['budgetsToTest']:
            for sr in experimentParams['sampleRatesToTest']:
                for al in experimentParams['ALToTest']:
                    
                    #Construct matrix
                    featureForClustering = baseData[mf].to_numpy()
                    adjMx = constructAdjMx(k,euclidPath,m,featureForClustering,oaIndex)
                    edgeIndexNp,edgeWeightsNp = loadAdj(adjMx,oaIndex)
                    
                    #Budget
                    b = int(len(baseData) * pb)
                    
                    #Construct training matrices
                    x, y, ySample, testMask, trainMask, seedMask, seedTrainMask, baseData, numSPQ, numFullSample, scalerY, scalerYSample = getTestTrainingData(baseData,pb,al,oaIndex,ss,dbLoc,poiInd,s,sr,mf,b)
                    
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
                        timeTried = 0
                        proceedMLP = False
                        while proceedMLP == False:
                            predVector, infTime, losses = MLPRegression(x,y,trainMask,testMask,hiddenMLP,epochsMLP, device)
                            timeTried += 1
                            if float(losses[-1].cpu().detach().numpy()) / float(losses[0].cpu().detach().numpy()) < 0.95:
                                proceedMLP = True
                            if timeTried == 10:
                                proceedMLP = True
                                experimentsTimeOut.append(expNum)
                        print()
                        print('Evaluating MLP')
                        absError,absErrorPcnt,jainActual,jainPred,jainsError,correation,corrConfidence,baseData = getPerformanceMetrics(testMask,scalerY,predVector,baseData,y,shpFileLoc,trainMask,poiLonLat,ymlFile,expNum)
                        writeResults(expNum,method,p, s, pb, sr, ss, al, absError,absErrorPcnt,jainActual,jainPred,jainsError,correation,corrConfidence,infTime,numSPQ,resultsFileName,baseData,ymlFile)
                    
                    if experimentParams['modelsToRun']['GNNSimple']:
                        #GNN Simple
                        expNum += 1
                        print('Experiment : ' + str(expNum))
                        method = 'GNN-Simple'
                        timeTried = 0
                        proceedGNNSimpl = False
                        while proceedGNNSimpl == False:
                            predVector, infTime, losses = GNNSimple(x,ySample,device,edgeIndexNp,edgeWeightsNp,hidden1GNN,hidden2GNN,epochsGNN,trainMask,testMask)
                            timeTried += 1
                            if float(losses[-1]) / float(losses[0]) < 0.95:
                                proceedGNNSimpl = True
                            if timeTried == 10:
                                proceedGNNSimpl = True
                                experimentsTimeOut.append(expNum)
                        print()
                        print('Evaluating GNN Simple')
                        absError,absErrorPcnt,jainActual,jainPred,jainsError,correation,corrConfidence,baseData = getPerformanceMetrics(testMask,scalerY,predVector,baseData,y,shpFileLoc,trainMask,poiLonLat,ymlFile,expNum)
                        writeResults(expNum,method,p, s, pb, sr, ss, al, absError,absErrorPcnt,jainActual,jainPred,jainsError,correation,corrConfidence,infTime,numSPQ,resultsFileName,baseData,ymlFile)
                    if experimentParams['modelsToRun']['GNNSeeds']:
                        for ss in experimentParams['seedSplitsToTest']:
                            #Construct training matrices
                            x, y, ySample, testMask, trainMask, seedMask, seedTrainMask, baseData, numSPQ, numFullSample, scalerY, scalerYSample = getTestTrainingData(baseData,pb,al,oaIndex,ss,dbLoc,poiInd,s,sr,mf,b)
                            
                            #Construct matrix
                            featureForClustering = baseData[mf].to_numpy()
                            adjMx = constructAdjMx(k,euclidPath,m,featureForClustering,oaIndex)
                            edgeIndexNp,edgeWeightsNp = loadAdj(adjMx,oaIndex)
                            
                            #GNN with Seeds
                            expNum += 1
                            print('Experiment : ' + str(expNum))
                            method = 'GNN-Seeds'
                            timeTried = 0
                            _x = appendPredictedCostToFeatures(baseData,seedMask,mfAcc,x,target='sampleAccessCost')
                            proceedGNNSeeds = False
                            while proceedGNNSeeds == False:
                                predVector, infTime, losses = GNNSimple(_x,ySample,device,edgeIndexNp,edgeWeightsNp,hidden1GNN,hidden2GNN,epochsGNN,seedTrainMask,testMask)
                                timeTried += 1
                                if float(losses[-1]) / float(losses[0]) < 0.95:
                                    proceedGNNSeeds = True
                                if timeTried == 10:
                                    proceedGNNSeeds = True
                                    experimentsTimeOut.append(expNum)
                            print()
                            print('Evaluating GNN Seeds')
                            absError,absErrorPcnt,jainActual,jainPred,jainsError,correation,corrConfidence,baseData = getPerformanceMetrics(testMask,scalerY,predVector,baseData,y,shpFileLoc,trainMask,poiLonLat,ymlFile,expNum)
                            writeResults(expNum,method,p, s, pb, sr, ss, al, absError,absErrorPcnt,jainActual,jainPred,jainsError,correation,corrConfidence,infTime,numSPQ,resultsFileName,baseData,ymlFile)
                    if experimentParams['modelsToRun']['COREG']:
                        expNum += 1
                        print('Experiment : ' + str(expNum))
                        method = 'COREG'
                        
                        t0 = time.time()
                        coregTrainer = CoregTrainer(data_dir,results_dir,num_train,num_trials,x,y,trainMask,testMask)
                        coregTrainer.run_trials()
                        predVector = coregTrainer.test_hat
                        t1 = time.time()
                        infTime = t1 - t0
                        
                        absError,absErrorPcnt,jainActual,jainPred,jainsError,correation,corrConfidence,baseData = getPerformanceMetrics(testMask,scalerY,predVector,baseData,y,shpFileLoc,trainMask,poiLonLat,ymlFile,expNum)
                        writeResults(expNum,method,p, s, pb, sr, ss, al, absError,absErrorPcnt,jainActual,jainPred,jainsError,correation,corrConfidence,infTime,numSPQ,resultsFileName,baseData,ymlFile)
                    if experimentParams['modelsToRun']['VAT']:
                        expNum += 1
                        print('Experiment : ' + str(expNum))
                        method = 'VAT'
                        
                        val = int(b*0.1)
                        valRecords = random.sample(range(b), val)
                        valMask = []
                        valTestMask = []

                        for i in range(b):
                            if i in valRecords:
                                valMask.append(True)
                                valTestMask.append(False)
                            else:
                                valTestMask.append(True)
                                valMask.append(False)
                                
                        t0 = time.time()
                        vatTrainer = VATTrainer(data_dir,results_dir,num_train,max_iters,num_trials,x, y, trainMask, testMask , valMask, valTestMask, device == 'cuda', num_unlabeled = len(baseData) - b,batch_unlabeled = len(baseData) - b)
                        vatTrainer.run_trials()
                        predVector = vatTrainer.model.forward(x[testMask]).eval()
                        t1 = time.time()
                        infTime = t1 - t0
                        
                        absError,absErrorPcnt,jainActual,jainPred,jainsError,correation,corrConfidence,baseData = getPerformanceMetrics(testMask,scalerY,predVector,baseData,y,shpFileLoc,trainMask,poiLonLat,ymlFile,expNum)
                        writeResults(expNum,method,p, s, pb, sr, ss, al, absError,absErrorPcnt,jainActual,jainPred,jainsError,correation,corrConfidence,infTime,numSPQ,resultsFileName,baseData,ymlFile)

print('The following experiments timed out:')
print(experimentsTimeOut)                        

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


# expNum += 1
# print('Experiment : ' + str(expNum))
# method = 'Regr-MLP'
# proceedMLP = False
# while proceedMLP == False:
#     predVector, infTime, losses = MLPRegression(x,y,trainMask,testMask,hiddenMLP,epochsMLP, device)                                
#     if float(losses[-1].cpu().detach().numpy()) / float(losses[0].cpu().detach().numpy()) < 0.95:
#         proceedMLP = True
# print()
# print('Evaluating MLP')
# absError,absErrorPcnt,jainActual,jainPred,jainsError,correation,corrConfidence,baseData = getPerformanceMetrics(testMask,scalerY,predVector,baseData,y,shpFileLoc,trainMask,poiLonLat,ymlFile,expNum)
# writeResults(expNum,method,p, s, pb, sr, ss, al, absError,absErrorPcnt,jainActual,jainPred,jainsError,correation,corrConfidence,infTime,numSPQ,resultsFileName,baseData,ymlFile)

# #%%

# #GNN Simple
# expNum += 1
# print('Experiment : ' + str(expNum))
# method = 'GNN-Simple'
# proceedGNNSimpl = False
# while proceedGNNSimpl == False:
#     predVector, infTime, losses = GNNSimple(x,ySample,device,edgeIndexNp,edgeWeightsNp,hidden1GNN,hidden2GNN,epochsGNN,trainMask,testMask)
#     print(float(losses[-1]) / float(losses[0]))
#     if float(losses[-1]) / float(losses[0]) < 0.95:
#         proceedGNNSimpl = True
# print()
# print('Evaluating GNN Simple')
# absError,absErrorPcnt,jainActual,jainPred,jainsError,correation,corrConfidence,baseData = getPerformanceMetrics(testMask,scalerY,predVector,baseData,y,shpFileLoc,trainMask,poiLonLat,ymlFile,expNum)
# writeResults(expNum,method,p, s, pb, sr, ss, al, absError,absErrorPcnt,jainActual,jainPred,jainsError,correation,corrConfidence,infTime,numSPQ,resultsFileName,baseData,ymlFile)

# #%%
# #GNN with Seeds
# #Construct matrix
# featureForClustering = baseData[mf].to_numpy()
# adjMx = constructAdjMx(k,euclidPath,m,featureForClustering,oaIndex)
# edgeIndexNp,edgeWeightsNp = loadAdj(adjMx,oaIndex)

# #GNN with Seeds
# expNum += 1
# print('Experiment : ' + str(expNum))
# method = 'GNN-Seeds'
# _x = appendPredictedCostToFeatures(baseData,seedMask,mfAcc,x,target='sampleAccessCost')
# proceedGNNSeeds = False
# while proceedGNNSeeds == False:
#     predVector, infTime, losses = GNNSimple(_x,ySample,device,edgeIndexNp,edgeWeightsNp,hidden1GNN,hidden2GNN,epochsGNN,seedTrainMask,testMask)
#     if float(losses[-1]) / float(losses[0]) < 0.95:
#         proceedGNNSeeds = True
# print()
# print('Evaluating GNN Seeds')
# absError,absErrorPcnt,jainActual,jainPred,jainsError,correation,corrConfidence,baseData = getPerformanceMetrics(testMask,scalerY,predVector,baseData,y,shpFileLoc,trainMask,poiLonLat,ymlFile,expNum)
# writeResults(expNum,method,p, s, pb, sr, ss, al, absError,absErrorPcnt,jainActual,jainPred,jainsError,correation,corrConfidence,infTime,numSPQ,resultsFileName,baseData,ymlFile)