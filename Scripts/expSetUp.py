import numpy as np
import random
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import sqlite3
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from Scripts.activeLearningStrategies import basicClustering, distCluster, degreeCentrality, eigenCentrality

#%%

# def getTestTrainingData(p,s,inputData,oaIndex,probeBudget,sampleRate,seedSplit,features,database,al,target='avgAccessCost'):
    
#     #Randomly select nodes for probe split
#     b = int(len(inputData) * probeBudget)
    
#     #Random
#     if al == 0:
#         trainRecords = random.sample(range(len(inputData)), b)
#     #basic clustering
#     elif al == 1:
#         trainRecords = basicClustering(b,inputData,oaIndex)
#     #dist clustering
#     elif al == 2:
#         trainRecords = distCluster(b,inputData)
#     #TODO: consider if we can take this out of this function and put it into a later function
#     #Randomly select seed records from probe
#     seefCutOff = int(b * seedSplit)
#     seedRecords = random.sample(trainRecords, seefCutOff)
    
#     # Get y based on sampling rate
    
#     #Get all trips for given POI type
#     # cnx = sqlite3.connect(dbLoc)
#     # sqlQuery = 'select a.oa_id, a.trip_id, b.stratum from otp_trips as a left join trip_strata as b on a.trip_id = b.trip_id where oa_id in {} and poi_id in (select distinct poi_id from poi where type = "{}");'.format(tuple(list(set(oaIndex))),p)
#     # allPOItrips = pd.read_sql_query(sqlQuery, cnx)
    
#     sqlQuery = "select a.oa_id, a.trip_id, b.stratum from model_may2022.otp_trips as a left join model_may2022.trips as b on a.trip_id = b.trip_id where a.oa_id in {} and a.poi_id in (select distinct id from semantic.poi where type = '{}');".format(tuple(list(set(oaIndex))),p)
#     allPOItrips = database.execute_sql(sqlQuery, False, True)

#     queryTripIds = []

#     #Randomly select trips from within the time stratum to sample the access cost
#     for oa in oaIndex:
#         oaTrips = allPOItrips[(allPOItrips['oa_id'] == oa) & (allPOItrips['stratum'] == s)]
#         probeCutOff = int(len(oaTrips) * sampleRate)
#         probeRecords = random.sample(range(len(oaTrips)), probeCutOff)
#         queryTripIds = queryTripIds + list(oaTrips['trip_id'].iloc[probeRecords])
    
#     #Get trip details
#     # sqlQuery = 'select total_time, initial_wait_time - 3600 as initial_wait_corrected, transit_time, fare, num_transfers, trip_id from otp_results where trip_id in {}'.format(tuple(list(set(queryTripIds))))
#     # tripsResults = pd.read_sql_query(sqlQuery, cnx)
    
#     #Get trip details
#     sqlQuery = 'select total_time, initial_wait_time - 3600 as initial_wait_corrected, transit_time, fare, num_transfers, trip_id from results_may2022.results_full where trip_id in {}'.format(tuple(list(set(queryTripIds))))
#     tripsResults = database.execute_sql(sqlQuery, False, True)
    
#     #Calculate access costs
#     tripsResults['sampleAccessCost'] = (( 1.5 * (tripsResults['total_time'] + tripsResults['initial_wait_corrected'])) - (0.5 * tripsResults['transit_time']) + ((tripsResults['fare'] * 3600) / 6.7) + (10 * tripsResults['num_transfers']) ) / 60
#     tripsResults = tripsResults.merge(allPOItrips[['oa_id','trip_id']],left_on = 'trip_id', right_on = 'trip_id', how = 'left')
#     inputData['sampleAccessCost'] = inputData['oa_id'].map(tripsResults.groupby('oa_id')['sampleAccessCost'].mean().to_dict()) 

#     # Normalize features
#     scalerX = MinMaxScaler()
#     scalerX.fit(np.array(inputData[features]))
#     x = scalerX.transform(inputData[features])
#     y = np.array(inputData['avgAccessCost'])
#     ySample = np.array(inputData['sampleAccessCost'])
    
#     #Generate test and training masks
#     testMask = []
#     trainMask = []

#     for i in range(len(inputData)):
#         if i in trainRecords:
#             testMask.append(False)
#             trainMask.append(True)
#         else:
#             testMask.append(True)
#             trainMask.append(False)

#     seedMask = []

#     for i in range(len(inputData)):
#         if i in seedRecords:
#             seedMask.append(True)
#         else:
#             seedMask.append(False)

#     seedTrainMask = []

#     for i in range(len(inputData)):
#         if (i in trainRecords) and (i not in seedRecords):
#             seedTrainMask.append(True)
#         else:
#             seedTrainMask.append(False)
            
#     return x, y, ySample, testMask, trainMask, seedMask, seedTrainMask, inputData, tripsResults.groupby('oa_id').size()[trainMask].sum(), len(tripsResults)

#%% Get predicted access cost based on probe data

def appendPredictedCostToFeatures(inputData,mask,features,x,target='sampleAccessCost'):
    #Train regression model on seed data to predict an access cost
    indVars = inputData[features][mask]
    depVars = inputData[target][mask]
    model = sm.OLS(depVars, indVars).fit()
    
    predictedAccessCost = []

    for i,r in inputData.iterrows():
        if mask[i]:
            predictedAccessCost.append(model.predict(r[features]).values[0])
        else:
            predictedAccessCost.append(r[target])        

    inputData['predictedScore'] = predictedAccessCost
    
    scalerPS = MinMaxScaler()
    scalerPS.fit(np.array(inputData['predictedScore']).reshape(-1, 1))
    predScore = scalerPS.transform(np.array(inputData['predictedScore']).reshape(-1, 1))
    
    #Append new feater onto x
    return np.c_[x, predScore]

#%%

#Function - Graph Construction

#EuclidMinMax
#EuclidGaussian
#K-Means Binary
#K-Means Euclidean

#%%

#Given base data and euclid matrix



def constructAdjMx(k,euclidPath,mxType,oaFeatureVec,oaIndex):

    euclidMx = np.load(euclidPath)
    
    #Euclidean with gaussian normalisation
    if mxType == 0:
    
        normalized_k = k
        adjMxFlat = euclidMx[~np.isinf(euclidMx)].flatten()
        std = adjMxFlat.std()
        
        adjMx = np.exp(-np.square(euclidMx / std))
        
        adjMx[adjMx < normalized_k] = 0
        adjMx[adjMx >= 1] = 0
    
    #Euclidean with min/man normalisation
    elif mxType == 1:
    
    
        adjMx = 1 - (euclidMx - euclidMx.min()) / (euclidMx.max() - euclidMx.min())
    
    
    elif mxType == 2:
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(oaFeatureVec)
        
        nClusters = int(scaled_features.shape[0]/20)
        
        kmeans = KMeans(init="random",n_clusters=nClusters,n_init=10,max_iter=500,random_state=42)
        kmeans.fit(scaled_features)
        
        clusterLabels = kmeans.labels_
        
        adjMx = np.zeros((len(oaIndex),len(oaIndex)))
        
        i = 0
        for oaI in oaIndex:
            j = 0
            for oaJ in oaIndex:
                if oaI == oaJ:
                    adjMx[i,j] = 1
                else:
                    if clusterLabels[i] == clusterLabels[j]:
                        adjMx[i,j] = 1
                    else:
                        adjMx[i,j] = 0
                j += 1
            i += 1
        
    
    elif mxType == 3:
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(oaFeatureVec)
        
        nClusters = int(scaled_features.shape[0]/40)
        
        kmeans = KMeans(init="random",n_clusters=nClusters,n_init=10,max_iter=1000,random_state=42)
        kmeans.fit(scaled_features)
        
        clusterLabels = kmeans.labels_
        
        adjMx = np.zeros((len(oaIndex),len(oaIndex)))
        
        countI = 0
        for i in oaIndex:
            oaCluster = clusterLabels[countI] 
            clusterIndex = np.where(clusterLabels == oaCluster)[0]
            oaDistances = euclidMx[countI,clusterIndex]
            norm = 1 - (oaDistances - oaDistances.min()) / (oaDistances.max() - oaDistances.min())
            countJ = 0
            for j in clusterIndex:
                adjMx[countI,j] = norm[countJ]
                countJ += 1
            countI += 1

    return adjMx

#%%

def getTestTrainingData(baseData,pb,al,oaIndex,ss,dbLoc,poiInd,s,sr,mf,b):
    
    #Random
    if al == 0:
        trainRecords = random.sample(range(len(baseData)), b)
    #basic clustering
    elif al == 1:
        trainRecords = basicClustering(b,baseData,oaIndex)
    #dist clustering
    elif al == 2:
        trainRecords = distCluster(b,baseData)
    #Distance centrality
    elif al == 3:
        trainRecords = degreeCentrality(b,baseData)
    #Eigenvector Centrality
    elif al == 4:
        trainRecords = eigenCentrality(b,baseData)
    #Randomly select seed records from probe
    seefCutOff = int(b * ss)
    seedRecords = random.sample(trainRecords, seefCutOff)
    
    cnx = sqlite3.connect(dbLoc)
    sqlQuery = 'select oa_id, trip_id, stratum from trips where oa_id in {} and poi_id in {} and stratum = "{}";'.format(tuple(list(set(oaIndex))),tuple(list(set(poiInd.astype(str)))),s)
    allPOItrips = pd.read_sql_query(sqlQuery, cnx)
    
    queryTripIds = []
    
    #Randomly select trips from within the time stratum to sample the access cost
    for oa in oaIndex:
        oaTrips = allPOItrips[(allPOItrips['oa_id'] == oa)]
        probeCutOff = int(len(oaTrips) * sr)
        probeRecords = random.sample(range(len(oaTrips)), probeCutOff)
        queryTripIds = queryTripIds + list(oaTrips['trip_id'].iloc[probeRecords])
    
    #Get trip details
    sqlQuery = 'select total_time, initial_wait_corrected, transit_time, fare, num_transfers, trip_id from results_full where trip_id in {}'.format(tuple(list(set(queryTripIds))))
    tripsResults = pd.read_sql_query(sqlQuery, cnx)
    
    #Calculate access costs
    tripsResults['sampleAccessCost'] = (( 1.5 * (tripsResults['total_time'] + tripsResults['initial_wait_corrected'])) - (0.5 * tripsResults['transit_time']) + ((tripsResults['fare'] * 3600) / 6.7) + (10 * tripsResults['num_transfers']) ) / 60
    
    tripsResults = tripsResults.merge(allPOItrips[['oa_id','trip_id']],left_on = 'trip_id', right_on = 'trip_id', how = 'left')
    
    baseData['sampleAccessCost'] = baseData['oa_id'].map(tripsResults.groupby('oa_id')['sampleAccessCost'].mean().to_dict()) 
    
    # Normalize features
    scalerX = MinMaxScaler()
    x = scalerX.fit_transform(np.array(baseData[mf]))
    
    scalerY = MinMaxScaler()
    y = scalerY.fit_transform(np.array(baseData['avgAccessCost']).reshape(-1, 1)).squeeze()
    
    scalerYSample = MinMaxScaler()
    ySample = scalerYSample.fit_transform(np.array(baseData['sampleAccessCost']).reshape(-1, 1)).squeeze()
    
    
    #Generate test and training masks
    testMask = []
    trainMask = []
    
    for i in range(len(baseData)):
        if i in trainRecords:
            testMask.append(False)
            trainMask.append(True)
        else:
            testMask.append(True)
            trainMask.append(False)
    
    seedMask = []
    
    for i in range(len(baseData)):
        if i in seedRecords:
            seedMask.append(True)
        else:
            seedMask.append(False)
    
    seedTrainMask = []
    
    for i in range(len(baseData)):
        if (i in trainRecords) and (i not in seedRecords):
            seedTrainMask.append(True)
        else:
            seedTrainMask.append(False)
            
    return x, y, ySample, testMask, trainMask, seedMask, seedTrainMask, baseData, tripsResults.groupby('oa_id').size()[trainMask].sum(), len(tripsResults), scalerY, scalerYSample