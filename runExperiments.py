from Scripts.dataPrep import getBaseTrainingData, loadAdj
from Scripts.expSetUp import getTestTrainingData,appendPredictedCostToFeatures,constructAdjMx, getSampleRateTrips
from Scripts.methods import simpleSampling,OLSRegression,MLPRegression,GNNSimple
from Scripts.evaluation import getPerformanceMetrics, writeResults
from Scripts.util.database import Database
from Baselines.labelPropagation import *
from Baselines.meanTeacher import *
import sqlite3
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
import geopandas as gpd
import pandas as pd

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

def runLabelProp(x,y,trainMask,testMask,valTestMask,valMask,val):
    
    X_labeled = x[trainMask]
    X_unlabeled = x[testMask]
    #y_labeled = y[trainMask].reshape(-1, 1)
    #y_unlabeled = y[testMask].reshape(-1, 1)
    y_labeled = y[trainMask]
    y_unlabeled = y[testMask]
    X_test = X_labeled[valTestMask]
    y_test = y_labeled[valTestMask]
    X_val = X_labeled[valMask]
    y_val = y_labeled[valMask]
    
    fold_results = pd.DataFrame()
    fold = 0
    
    # search over sigma_2
    sigma_2s = np.linspace(0.8, 3.0, 5)
    val_losses = Parallel(n_jobs=5)(delayed(
        label_propagation_regression)(X_test, y_test, X_unlabeled, X_val, y_val, sigma_2)
        for sigma_2 in sigma_2s)
    
    
    best_idx = np.argmin(val_losses)
    
    best_val_loss = val_losses[best_idx]
    best_sigma_2 = sigma_2s[best_idx]
    fold_result_row = {
                       'fold': fold, 'best_val_loss': best_val_loss,
                       'best_sigma_2': best_sigma_2,
                       'sigma_2s': sigma_2s,
                       'val_losses': val_losses}
    fold_results = fold_results.append(fold_result_row, ignore_index=True)
    
    # test with the best
    val_loss, y_all = label_propagation_regression(X_test, y_test, X_unlabeled, X_val, y_val, best_sigma_2, True)
    
    return y_all[(b-val):-val]

print('--Modules Imported--')

#%% Import experiment yaml

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

area = experimentParams['geoAreas'][0]

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

# Get Geouits

# Read in base data - shapefile / OA info / OA index
wm_oas = gpd.read_file(shpFileLoc)
wm_oas = wm_oas[wm_oas['LAD11CD'] == experimentParams['geoAreas'][0]]
oa_info = pd.read_csv(oaInfoLoc)
oa_info = oa_info.merge(wm_oas[['OA11CD']], left_on = 'oa_id', right_on = 'OA11CD', how = 'inner')
oaIndex = list(oa_info['oa_id'])

adjMx = np.load('Data/adjMx/' + str(area) + '/adjMx.csv')
edgeIndexNp,edgeWeightsNp = loadAdj(adjMx,oaIndex)
cnx = sqlite3.connect(dbLoc)


#%%

expNum = 0
experimentsTimeOut = []
experimentsFailed = []
#Use placehold value in first instance
ss = experimentParams['seedSplitsToTest'][0]

for i in range(experimentParams['trials']):

    for p in experimentParams['POIsToTest']:
        for s in experimentParams['stratemsToTest']:
            #Get base training data (e.g., OAs with relevant features from POI and stratum)
            baseData, oaIndex, poiLonLat, poiInd = getBaseTrainingData(shpFileLoc,oaInfoLoc,s,p,dbLoc,wm_oas,oa_info,oaIndex)
            
            #Get trips for POIs and Stratum
            sqlQuery = 'select oa_id, trip_id, stratum from trips where oa_id in {} and poi_id in {} and stratum = "{}";'.format(tuple(list(set(oaIndex))),tuple(list(set(poiInd.astype(str)))),s)
            allPOItrips = pd.read_sql_query(sqlQuery, cnx)
            
            for sr in experimentParams['sampleRatesToTest']:
                
                x, y, ySample, baseData, tripsResults, numFullSample, scalerY, scalerYSample = getSampleRateTrips(oaIndex,allPOItrips,sr,cnx,baseData,mf)
                
                for pb in experimentParams['budgetsToTest']:
                    for al in experimentParams['ALToTest']:
                                            
                        #Budget
                        b = int(len(baseData) * pb)
                        
                        #Construct training matrices
                        # x, y, ySample, testMask, trainMask, seedMask, seedTrainMask, baseData, numSPQ, numFullSample, scalerY, scalerYSample = getTestTrainingData(baseData,pb,al,oaIndex,ss,dbLoc,poiInd,s,sr,mf,b,area)
                        
                        testMask, trainMask, seedMask, seedTrainMask, valMask, valTestMask, val, numSPQ = getTestTrainingData(al,baseData,b,oaIndex,mf,area,ss,tripsResults)
                        
                        
                        # val = int(b*0.1)
                        # valRecords = random.sample(range(b), val)
                        # valMask = []
                        # valTestMask = []

                        # for i in range(b):
                        #     if i in valRecords:
                        #         valMask.append(True)
                        #         valTestMask.append(False)
                        #     else:
                        #         valTestMask.append(True)
                        #         valMask.append(False)
                        
                        if experimentParams['modelsToRun']['OLS']:
                            #OLS Regression
                            expNum += 1
                            print('Experiment : ' + str(expNum))
                            method = 'Regr-OLS'
                            try:
                                predVector, infTime = OLSRegression(x,y,trainMask,testMask)
                                absError,absErrorPcnt,jainActual,jainPred,jainsError,correation,corrConfidence,baseData = getPerformanceMetrics(testMask,scalerY,predVector,baseData,y,shpFileLoc,trainMask,poiLonLat,ymlFile,expNum,area)
                                writeResults(expNum,method,p, s, pb, sr, ss, al, absError,absErrorPcnt,jainActual,jainPred,jainsError,correation,corrConfidence,infTime,numSPQ,resultsFileName,baseData,ymlFile)
                            except Exception as e:
                                print(e)
                                print('fail: ' + str(expNum))
                                experimentsFailed.append(expNum)
                        if experimentParams['modelsToRun']['MLP']:
                            expNum += 1
                            print('Experiment : ' + str(expNum))
                            method = 'Regr-MLP'
                            try:
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
                                absError,absErrorPcnt,jainActual,jainPred,jainsError,correation,corrConfidence,baseData = getPerformanceMetrics(testMask,scalerY,predVector,baseData,y,shpFileLoc,trainMask,poiLonLat,ymlFile,expNum,area)
                                writeResults(expNum,method,p, s, pb, sr, ss, al, absError,absErrorPcnt,jainActual,jainPred,jainsError,correation,corrConfidence,infTime,numSPQ,resultsFileName,baseData,ymlFile)
                            except Exception as e:
                                print(e)
                                print('fail: ' + str(expNum))
                                experimentsFailed.append(expNum)
                        if experimentParams['modelsToRun']['GNNSimple']:
                            #GNN Simple
                            expNum += 1
                            print('Experiment : ' + str(expNum))
                            method = 'GNN-Simple'
                            try:
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
                                absError,absErrorPcnt,jainActual,jainPred,jainsError,correation,corrConfidence,baseData = getPerformanceMetrics(testMask,scalerY,predVector,baseData,y,shpFileLoc,trainMask,poiLonLat,ymlFile,expNum,area)
                                writeResults(expNum,method,p, s, pb, sr, ss, al, absError,absErrorPcnt,jainActual,jainPred,jainsError,correation,corrConfidence,infTime,numSPQ,resultsFileName,baseData,ymlFile)
                            except Exception as e:
                                print(e)
                                print('fail: ' + str(expNum))
                                experimentsFailed.append(expNum)
                        if experimentParams['modelsToRun']['GNNSeeds']:
                            if len(experimentParams['seedSplitsToTest']) > 1:
                                for ss in experimentParams['seedSplitsToTest']:
                                    #Construct training matrices
                                    x, y, ySample, testMask, trainMask, seedMask, seedTrainMask, baseData, numSPQ, numFullSample, scalerY, scalerYSample = getTestTrainingData(baseData,pb,al,oaIndex,ss,dbLoc,poiInd,s,sr,mf,b,area)
                                    
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
                                    
                                    #Construct matrix
                                    featureForClustering = baseData[mf].to_numpy()
                                    adjMx = constructAdjMx(k,euclidPath,m,featureForClustering,oaIndex)
                                    edgeIndexNp,edgeWeightsNp = loadAdj(adjMx,oaIndex)
                                    
                                    #GNN with Seeds
                                    expNum += 1
                                    print('Experiment : ' + str(expNum))
                                    method = 'GNN-Seeds'
                                    try:
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
                                        absError,absErrorPcnt,jainActual,jainPred,jainsError,correation,corrConfidence,baseData = getPerformanceMetrics(testMask,scalerY,predVector,baseData,y,shpFileLoc,trainMask,poiLonLat,ymlFile,expNum,area)
                                        writeResults(expNum,method,p, s, pb, sr, ss, al, absError,absErrorPcnt,jainActual,jainPred,jainsError,correation,corrConfidence,infTime,numSPQ,resultsFileName,baseData,ymlFile)
                                    except Exception as e:
                                        print(e)
                                        print('fail: ' + str(expNum))
                                        experimentsFailed.append(expNum)
                            else:
                                #GNN with Seeds
                                expNum += 1
                                print('Experiment : ' + str(expNum))
                                method = 'GNN-Seeds'
                                try:
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
                                    absError,absErrorPcnt,jainActual,jainPred,jainsError,correation,corrConfidence,baseData = getPerformanceMetrics(testMask,scalerY,predVector,baseData,y,shpFileLoc,trainMask,poiLonLat,ymlFile,expNum,area)
                                    writeResults(expNum,method,p, s, pb, sr, ss, al, absError,absErrorPcnt,jainActual,jainPred,jainsError,correation,corrConfidence,infTime,numSPQ,resultsFileName,baseData,ymlFile)
                                except Exception as e:
                                    print(e)
                                    print('fail: ' + str(expNum))
                                    experimentsFailed.append(expNum)
                        if experimentParams['modelsToRun']['COREG']:
                            expNum += 1
                            print('Experiment : ' + str(expNum))
                            method = 'COREG'
                            try:
                                t0 = time.time()
                                coregTrainer = CoregTrainer(data_dir,results_dir,num_train,num_trials,x,y,trainMask,testMask)
                                coregTrainer.run_trials()
                                predVector = coregTrainer.test_hat
                                t1 = time.time()
                                infTime = t1 - t0
                                
                                absError,absErrorPcnt,jainActual,jainPred,jainsError,correation,corrConfidence,baseData = getPerformanceMetrics(testMask,scalerY,predVector,baseData,y,shpFileLoc,trainMask,poiLonLat,ymlFile,expNum,area)
                                writeResults(expNum,method,p, s, pb, sr, ss, al, absError,absErrorPcnt,jainActual,jainPred,jainsError,correation,corrConfidence,infTime,numSPQ,resultsFileName,baseData,ymlFile)
                            except Exception as e:
                                print(e)
                                print('fail: ' + str(expNum))
                                experimentsFailed.append(expNum)
                        if experimentParams['modelsToRun']['VAT']:
                            expNum += 1
                            print('Experiment : ' + str(expNum))
                            method = 'VAT'
                            try:
                                        
                                t0 = time.time()
                                vatTrainer = VATTrainer(data_dir,results_dir,num_train,max_iters,num_trials,x, y, trainMask, testMask , valMask, valTestMask, device == 'cuda', num_unlabeled = len(baseData) - b,batch_unlabeled = len(baseData) - b)
                                vatTrainer.run_trials()
                                predVector = vatTrainer.model.forward(x[testMask]).eval()
                                t1 = time.time()
                                infTime = t1 - t0
                                
                                absError,absErrorPcnt,jainActual,jainPred,jainsError,correation,corrConfidence,baseData = getPerformanceMetrics(testMask,scalerY,predVector,baseData,y,shpFileLoc,trainMask,poiLonLat,ymlFile,expNum,area)
                                writeResults(expNum,method,p, s, pb, sr, ss, al, absError,absErrorPcnt,jainActual,jainPred,jainsError,correation,corrConfidence,infTime,numSPQ,resultsFileName,baseData,ymlFile)
                            except Exception as e:
                                print(e)
                                print('fail: ' + str(expNum))
                                experimentsFailed.append(expNum)

                        if experimentParams['modelsToRun']['LabProp']:
                            expNum += 1
                            print('Experiment : ' + str(expNum))
                            method = 'LabProp'
                            try:
                                t0 = time.time()
                                predVector = runLabelProp(x,y,trainMask,testMask,valTestMask,valMask,val)
                                t1 = time.time()
                                infTime = t1 - t0
                                absError,absErrorPcnt,jainActual,jainPred,jainsError,correation,corrConfidence,baseData = getPerformanceMetrics(testMask,scalerY,predVector,baseData,y,shpFileLoc,trainMask,poiLonLat,ymlFile,expNum,area)
                                writeResults(expNum,method,p, s, pb, sr, ss, al, absError,absErrorPcnt,jainActual,jainPred,jainsError,correation,corrConfidence,infTime,numSPQ,resultsFileName,baseData,ymlFile)
                            except Exception as e:
                                print(e)
                                print('fail: ' + str(expNum))
                                experimentsFailed.append(expNum)


                        if experimentParams['modelsToRun']['meanTeacher']:
                            expNum += 1
                            print('Experiment : ' + str(expNum))
                            method = 'meanTeacher'
                            try:
                                t0 = time.time()
                                tf.reset_default_graph()
                                
                                X_labeled = x[trainMask]
                                X_unlabeled = x[testMask]
                                #y_labeled = y[trainMask].reshape(-1, 1)
                                #y_unlabeled = y[testMask].reshape(-1, 1)
                                y_labeled = y[trainMask]
                                y_unlabeled = y[testMask]
                                X_test = X_labeled[valTestMask]
                                y_test = y_labeled[valTestMask]
                                X_val = X_labeled[valMask]
                                y_val = y_labeled[valMask]
                                
                                test_loss, best_val_loss, predictionStudent = optimize_mean_teacher(X_labeled, y_labeled, X_unlabeled, y_unlabeled,X_val, y_val,X_test, y_test)
                                tf.reset_default_graph()
                                t1 = time.time()
                                infTime = t1 - t0
                                absError,absErrorPcnt,jainActual,jainPred,jainsError,correation,corrConfidence,baseData = getPerformanceMetrics(testMask,scalerY,predVector,baseData,y,shpFileLoc,trainMask,poiLonLat,ymlFile,expNum,area)
                                writeResults(expNum,method,p, s, pb, sr, ss, al, absError,absErrorPcnt,jainActual,jainPred,jainsError,correation,corrConfidence,infTime,numSPQ,resultsFileName,baseData,ymlFile)
                            except Exception as e:
                                print(e)
                                print('fail: ' + str(expNum))
                                experimentsFailed.append(expNum)
print('The following experiments timed out:')
print(experimentsTimeOut)
print()
print('The following experiments failed:')
print(experimentsFailed)

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