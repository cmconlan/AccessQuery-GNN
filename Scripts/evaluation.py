import numpy as np
from scipy.stats.stats import pearsonr
import csv
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd

#%%

#Get performance statistics
#Output performance statistic to central repository
#Plot access on map and output

def performanceMetrics(predVector,inputData,testMask,y,shpFileLoc,oaInfoLoc,runRef,expNum, trainMask, poiLonLat, x):
    
    predInd = 0
    predicted = []

    for i in range(len(testMask)):
        if testMask[i]:
            predicted.append(predVector[predInd])
            predInd += 1
        else:
            predicted.append(inputData['avgAccessCost'].values[i])

    predicted = np.array(predicted)
    
    # Get absolute error
    error = abs(y - predicted)
    absError = error.mean()
    
    # Get percentage error
    errorPct = (abs(y - predicted)) / y
    absErrorPcnt = errorPct.mean()
    
    #Jains error
    jainActual = (y.sum() ** 2) / ((y*y).sum() * y.shape[0])
    jainPred = (predicted.sum() ** 2) / ((predicted*predicted).sum() * predicted.shape[0])
    jainsError = jainActual - jainPred
    
    #Get correlation coefficient
    correation, corrConfidence = pearsonr(predicted,y)
    
    #OUTPUT PLOTS
    
    #add predicted to input data
    inputData['predictedAccessCost'] = predicted
    inputData['absError'] = error
    inputData['absErrorPcnt'] = errorPct
    
    #Read in oa info
    oa_info = pd.read_csv(oaInfoLoc)
    
    #read in shape file
    wm_oas = gpd.read_file(shpFileLoc)
    wm_oas = wm_oas[wm_oas['LAD11CD'] == 'E08000026']

    inputData = inputData.merge(oa_info[['oa_id','oa_lat','oa_lon']],left_on = 'oa_id',right_on = 'oa_id',how = 'left')
    wm_oas = wm_oas.merge(inputData[['oa_id','avgAccessCost','predictedAccessCost']], left_on = 'OA11CD', right_on = 'oa_id', how = 'left')
    
    
    fig, axs = plt.subplots(8,2,figsize=(15,40))
    
    #Row 1
    axs[0,0].hist(error, bins=20)
    axs[0,0].set_title('Histogram of errors (at OA level)')
    axs[0,0].set_xlabel('Error')
    
    axs[0,1].hist(errorPct, bins=20)
    axs[0,1].set_title('Histogram of error percent (at oa level)')
    axs[0,1].set_xlabel('Error Percent')
    
    #Row 2
    for oa_i,oa_r in inputData[trainMask].iterrows():
        axs[1,0].scatter(x = oa_r['oa_lon'], y = oa_r['oa_lat'], s = 30, c = 'fuchsia', alpha = 0.5, marker = "p")
        
    inputData.plot.scatter(x = 'oa_lon', y = 'oa_lat', c = 'absError', colormap='viridis', ax = axs[1,0], s=6)
    
    for poi in poiLonLat:
        if  poi[0] >= min(inputData['oa_lon']) and poi[0] <= max(inputData['oa_lon']) and poi[1] >= min(inputData['oa_lat']) and poi[1] <= max(inputData['oa_lat']):
            axs[1,0].scatter(x = poi[0], y = poi[1], s = 6, c = 'red', alpha = 0.4, marker = "^")
    
    axs[1,0].set_title('OAs plotted with error on colour scale')
    
    for oa_i,oa_r in inputData[trainMask].iterrows():
        axs[1,1].scatter(x = oa_r['oa_lon'], y = oa_r['oa_lat'], s = 30, c = 'fuchsia', alpha = 0.5, marker = "p")
    
    inputData.plot.scatter(x = 'oa_lon', y = 'oa_lat', c = 'absErrorPcnt', colormap='viridis', ax = axs[1,1], s=6)
    for poi in poiLonLat:
        if  poi[0] >= min(inputData['oa_lon']) and poi[0] <= max(inputData['oa_lon']) and poi[1] >= min(inputData['oa_lat']) and poi[1] <= max(inputData['oa_lat']):
            axs[1,1].scatter(x = poi[0], y = poi[1], s = 6, c = 'red', alpha = 0.4, marker = "^")
    axs[1,1].set_title('OAs plotted with error percent on colour scale')
    
    
    #Row 3
    
    for oa_i,oa_r in inputData[trainMask].iterrows():
        axs[2,0].scatter(x = oa_r['oa_lon'], y = oa_r['oa_lat'], s = 30, c = 'fuchsia', alpha = 0.5, marker = "p")
        
    inputData.sample(frac = 0.4).plot.scatter(x = 'oa_lon', y = 'oa_lat', c = 'absError', colormap='viridis', ax = axs[2,0], s=6)
    
    for poi in poiLonLat:
        if  poi[0] >= min(inputData['oa_lon']) and poi[0] <= max(inputData['oa_lon']) and poi[1] >= min(inputData['oa_lat']) and poi[1] <= max(inputData['oa_lat']):
            axs[2,0].scatter(x = poi[0], y = poi[1], s = 6, c = 'red', alpha = 0.4, marker = "^")
    
    axs[2,0].set_title('OAs (sampled) plotted with error on colour scale')
    
    for oa_i,oa_r in inputData[trainMask].iterrows():
        axs[2,1].scatter(x = oa_r['oa_lon'], y = oa_r['oa_lat'], s = 30, c = 'fuchsia', alpha = 0.5, marker = "p")
    
    inputData.sample(frac = 0.4).plot.scatter(x = 'oa_lon', y = 'oa_lat', c = 'absErrorPcnt', colormap='viridis', ax = axs[2,1], s=6)
    for poi in poiLonLat:
        if  poi[0] >= min(inputData['oa_lon']) and poi[0] <= max(inputData['oa_lon']) and poi[1] >= min(inputData['oa_lat']) and poi[1] <= max(inputData['oa_lat']):
            axs[2,1].scatter(x = poi[0], y = poi[1], s = 6, c = 'red', alpha = 0.4, marker = "^")
    axs[2,1].set_title('OAs (sampled) plotted with error percent on colour scale')
    
    #Row 4
    wm_oas.plot(column='avgAccessCost', cmap='OrRd', scheme='quantiles', ax = axs[3,0])
    axs[3,0].set_title('Actual Access Costs')
    wm_oas.plot(column='predictedAccessCost', cmap='OrRd', scheme='quantiles', ax = axs[3,1])
    axs[3,1].set_title('Predicted Access Costs')
    
    
    #Row 5
    axs[4,0].scatter(x = x[:,0], y = inputData['absError'])
    axs[4,0].set_title('Scatter plot of error vs Avg Dist to POI')
    axs[4,1].scatter(x = x[:,0], y = inputData['absErrorPcnt'])
    axs[4,1].set_title('Scatter plot of error percent vs Avg Dist to POI')
    
    #Row 6
    axs[5,0].scatter(x = x[:,1], y = inputData['absError'])
    axs[5,0].set_title('Scatter plot of error vs OA Area')
    axs[5,1].scatter(x = x[:,1], y = inputData['absErrorPcnt'])
    axs[5,1].set_title('Scatter plot of error percent vs OA Area')
    
    #Row 7
    axs[6,0].scatter(x = x[:,2], y = inputData['absError'])
    axs[6,0].set_title('Scatter plot of error vs Distance to centre')
    axs[6,1].scatter(x = x[:,2], y = inputData['absErrorPcnt'])
    axs[6,1].set_title('Scatter plot of error percent vs Distance to centre')
    
    #Row 7
    axs[7,0].scatter(x = x[:,3], y = inputData['absError'])
    axs[7,0].set_title('Scatter plot of error vs population density')
    axs[7,1].scatter(x = x[:,3], y = inputData['absErrorPcnt'])
    axs[7,1].set_title('Scatter plot of error percent vs population density')

    plt.savefig('Results/Plots/'+runRef+'/'+str(expNum)+'.png', bbox_inches='tight')
    plt.cla()
    plt.close(fig)
    
    featureCorrelationsError = [pearsonr(x[:,0],inputData['absError'])[0],pearsonr(x[:,1],inputData['absError'])[0],pearsonr(x[:,2],inputData['absError'])[0],pearsonr(x[:,3],inputData['absError'])[0]]
    featureCorrelationsErrorPcnt = [pearsonr(x[:,0],inputData['absErrorPcnt'])[0],pearsonr(x[:,1],inputData['absErrorPcnt'])[0],pearsonr(x[:,2],inputData['absErrorPcnt'])[0],pearsonr(x[:,3],inputData['absErrorPcnt'])[0]]
    
    return absError,absErrorPcnt,jainActual,jainPred,jainsError,correation,corrConfidence, featureCorrelationsError, featureCorrelationsErrorPcnt, inputData

#%% add results to repository

def writeResults(expNum,method,p, s, probeBudget, sampleRate, seedSplit, aType, absError,absErrorPcnt,jainActual,jainPred,jainsError,correation,corrConfidence,featureCorrelationsError,featureCorrelationsErrorPcnt,infTime,numSPQ,resultsFileName,inputData,runRef):

    resultRow = [expNum, method,p, s, probeBudget, sampleRate, seedSplit, aType, absError,absErrorPcnt,jainActual,jainPred,jainsError,correation,corrConfidence,featureCorrelationsError[0],featureCorrelationsError[1],featureCorrelationsError[2],featureCorrelationsError[3],featureCorrelationsErrorPcnt[0],featureCorrelationsErrorPcnt[1],featureCorrelationsErrorPcnt[2],featureCorrelationsErrorPcnt[3],infTime,numSPQ]
    with open(resultsFileName,'a', newline='') as f:
        read_file = csv.writer(f)
        read_file.writerow(resultRow)

    inputData.to_csv('Results/Data/'+runRef+'/'+str(expNum)+'.csv')