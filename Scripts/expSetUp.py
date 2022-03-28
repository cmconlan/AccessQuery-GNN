import numpy as np
import random
from sklearn.preprocessing import MinMaxScaler
import sqlite3
import pandas as pd
import statsmodels.api as sm

#%%

def getTestTrainingData(p,s,inputData,oaIndex,dbLoc,probeBudget,sampleRate,seedSplit,features,target='avgAccessCost'):
    
    #Randomly select nodes for probe split
    trainCutOff = int(len(inputData) * probeBudget)
    trainRecords = random.sample(range(len(inputData)), trainCutOff)

    #TODO: consider if we can take this out of this function and put it into a later function
    #Randomly select seed records from probe
    seefCutOff = int(trainCutOff * seedSplit)
    seedRecords = random.sample(trainRecords, seefCutOff)
    
    # Get y based on sampling rate
    
    #Get all trips for given POI type
    cnx = sqlite3.connect(dbLoc)
    sqlQuery = 'select a.oa_id, a.trip_id, b.stratum from otp_trips as a left join trip_strata as b on a.trip_id = b.trip_id where oa_id in {} and poi_id in (select distinct poi_id from poi where type = "{}");'.format(tuple(list(set(oaIndex))),p)
    allPOItrips = pd.read_sql_query(sqlQuery, cnx)

    queryTripIds = []

    #Randomly select trips from within the time stratum to sample the access cost
    for oa in oaIndex:
        oaTrips = allPOItrips[(allPOItrips['oa_id'] == oa) & (allPOItrips['stratum'] == s)]
        probeCutOff = int(len(oaTrips) * sampleRate)
        probeRecords = random.sample(range(len(oaTrips)), probeCutOff)
        queryTripIds = queryTripIds + list(oaTrips['trip_id'].iloc[probeRecords])
    
    #Get trip details
    sqlQuery = 'select total_time, initial_wait_time - 3600 as initial_wait_corrected, transit_time, fare, num_transfers, trip_id from otp_results where trip_id in {}'.format(tuple(list(set(queryTripIds))))
    tripsResults = pd.read_sql_query(sqlQuery, cnx)
    
    #Calculate access costs
    tripsResults['sampleAccessCost'] = (( 1.5 * (tripsResults['total_time'] + tripsResults['initial_wait_corrected'])) - (0.5 * tripsResults['transit_time']) + ((tripsResults['fare'] * 3600) / 6.7) + (10 * tripsResults['num_transfers']) ) / 60
    tripsResults = tripsResults.merge(allPOItrips[['oa_id','trip_id']],left_on = 'trip_id', right_on = 'trip_id', how = 'left')
    inputData['sampleAccessCost'] = inputData['oa_id'].map(tripsResults.groupby('oa_id')['sampleAccessCost'].mean().to_dict()) 

    # Normalize features
    scalerX = MinMaxScaler()
    scalerX.fit(np.array(inputData[features]))
    x = scalerX.transform(inputData[features])
    y = np.array(inputData['avgAccessCost'])
    ySample = np.array(inputData['sampleAccessCost'])
    
    #Generate test and training masks
    testMask = []
    trainMask = []

    for i in range(len(inputData)):
        if i in trainRecords:
            testMask.append(False)
            trainMask.append(True)
        else:
            testMask.append(True)
            trainMask.append(False)

    seedMask = []

    for i in range(len(inputData)):
        if i in seedRecords:
            seedMask.append(True)
        else:
            seedMask.append(False)

    seedTrainMask = []

    for i in range(len(inputData)):
        if (i in trainRecords) and (i not in seedRecords):
            seedTrainMask.append(True)
        else:
            seedTrainMask.append(False)
            
    return x, y, ySample, testMask, trainMask, seedMask, seedTrainMask, inputData, tripsResults.groupby('oa_id').size()[trainMask].sum()



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
