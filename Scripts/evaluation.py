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

def getPerformanceMetrics(testMask,scalerY,predVector,baseData,y,shpFileLoc,trainMask,poiLonLat,ymlFile,expNum, area):

    predInd = 0
    predicted = []
    
    for i in range(len(testMask)):
        if testMask[i]:
            predicted.append(scalerY.inverse_transform(predVector[predInd].reshape(1, -1))[0][0])
            predInd += 1
        else:
            predicted.append(baseData['avgAccessCost'].values[i])
    
    predicted = np.array(predicted)
    
    y_hat = np.squeeze(scalerY.inverse_transform(y.reshape(1, -1)),axis = 0)
    
    # Get absolute error
    error = abs(y_hat - predicted)
    absError = error.mean()
    
    # Get percentage error
    errorPct = (abs(y_hat - predicted)) / y_hat
    absErrorPcnt = errorPct.mean()
    
    #Jains error
    jainActual = (y_hat.sum() ** 2) / ((y_hat*y_hat).sum() * y_hat.shape[0])
    jainPred = (predicted.sum() ** 2) / ((predicted*predicted).sum() * predicted.shape[0])
    jainsError = jainActual - jainPred
    
    #Get correlation coefficient
    correation, corrConfidence = pearsonr(predicted,y_hat)
    
    #OUTPUT PLOTS
    
    #add predicted to input data
    baseData['predictedAccessCost'] = predicted
    baseData['absError'] = error
    baseData['absErrorPcnt'] = errorPct
        
    #read in shape file
    # wm_oas = gpd.read_file(shpFileLoc)
    # wm_oas = wm_oas[wm_oas['LAD11CD'] == area]
    
    # wm_oas = wm_oas.merge(baseData[['oa_id','avgAccessCost','predictedAccessCost']], left_on = 'OA11CD', right_on = 'oa_id', how = 'left')
    
    # fig, axs = plt.subplots(3,2,figsize=(10,12))
    
    # #Row 2
    # for oa_i,oa_r in baseData[trainMask].iterrows():
    #     axs[0,0].scatter(x = oa_r['oa_lon'], y = oa_r['oa_lat'], s = 30, c = 'fuchsia', alpha = 0.5, marker = "p")
        
    # baseData.plot.scatter(x = 'oa_lon', y = 'oa_lat', c = 'absError', colormap='viridis', ax = axs[0,0], s=6)
    
    # for poi in poiLonLat:
    #     if  poi[0] >= min(baseData['oa_lon']) and poi[0] <= max(baseData['oa_lon']) and poi[1] >= min(baseData['oa_lat']) and poi[1] <= max(baseData['oa_lat']):
    #         axs[0,0].scatter(x = poi[0], y = poi[1], s = 6, c = 'red', alpha = 0.4, marker = "^")
    
    # axs[0,0].set_title('OAs plotted with error on colour scale')
    
    # for oa_i,oa_r in baseData[trainMask].iterrows():
    #     axs[0,1].scatter(x = oa_r['oa_lon'], y = oa_r['oa_lat'], s = 30, c = 'fuchsia', alpha = 0.5, marker = "p")
    
    # baseData.plot.scatter(x = 'oa_lon', y = 'oa_lat', c = 'absErrorPcnt', colormap='viridis', ax = axs[0,1], s=6)
    # for poi in poiLonLat:
    #     if  poi[0] >= min(baseData['oa_lon']) and poi[0] <= max(baseData['oa_lon']) and poi[1] >= min(baseData['oa_lat']) and poi[1] <= max(baseData['oa_lat']):
    #         axs[0,1].scatter(x = poi[0], y = poi[1], s = 6, c = 'red', alpha = 0.4, marker = "^")
    # axs[0,1].set_title('OAs plotted with error percent on colour scale')
    
    
    # #Row 3
    
    # for oa_i,oa_r in baseData.iterrows():
    #     axs[1,0].scatter(x = oa_r['oa_lon'], y = oa_r['oa_lat'], s = 10, c = 'fuchsia', marker = ".")
    
    # for poi in poiLonLat:
    #     if  poi[0] >= min(baseData['oa_lon']) and poi[0] <= max(baseData['oa_lon']) and poi[1] >= min(baseData['oa_lat']) and poi[1] <= max(baseData['oa_lat']):
    #         axs[1,0].scatter(x = poi[0], y = poi[1], s = 40, c = 'blue', marker = "^")
    
    # for oa_i,oa_r in baseData[trainMask].iterrows():
    #     axs[1,0].scatter(x = oa_r['oa_lon'], y = oa_r['oa_lat'], s = 30, alpha = 0.6,c = 'green', marker = "o")
    
    # axs[1,0].set_title('Map of OAs, seed OAs and POI')
    
    # for oa_i,oa_r in baseData[trainMask].iterrows():
    #     axs[1,1].scatter(x = oa_r['oa_lon'], y = oa_r['oa_lat'], s = 30, c = 'fuchsia', alpha = 0.5, marker = "p")
    
    # baseData.sample(frac = 0.4).plot.scatter(x = 'oa_lon', y = 'oa_lat', c = 'absErrorPcnt', colormap='viridis', ax = axs[1,1], s=6)
    # for poi in poiLonLat:
    #     if  poi[0] >= min(baseData['oa_lon']) and poi[0] <= max(baseData['oa_lon']) and poi[1] >= min(baseData['oa_lat']) and poi[1] <= max(baseData['oa_lat']):
    #         axs[1,1].scatter(x = poi[0], y = poi[1], s = 6, c = 'red', alpha = 0.4, marker = "^")
    # axs[1,1].set_title('OAs (sampled) plotted with error percent on colour scale')
    
    # #Row 4
    # wm_oas.plot(column='avgAccessCost', cmap='OrRd', scheme='quantiles', ax = axs[2,0])
    # for poi in poiLonLat:
    #     if  poi[0] >= min(baseData['oa_lon']) and poi[0] <= max(baseData['oa_lon']) and poi[1] >= min(baseData['oa_lat']) and poi[1] <= max(baseData['oa_lat']):
    #         axs[2,0].scatter(x = poi[0], y = poi[1], s = 10, alpha = 0.6, c = 'blue', marker = "^")
    # axs[2,0].set_title('Actual Access Costs')
    # wm_oas.plot(column='predictedAccessCost', cmap='OrRd', scheme='quantiles', ax = axs[2,1])
    # for poi in poiLonLat:
    #     if  poi[0] >= min(baseData['oa_lon']) and poi[0] <= max(baseData['oa_lon']) and poi[1] >= min(baseData['oa_lat']) and poi[1] <= max(baseData['oa_lat']):
    #         axs[2,1].scatter(x = poi[0], y = poi[1], s = 10, alpha = 0.6, c = 'blue', marker = "^")
    # axs[2,1].set_title('Predicted Access Costs')
    
    # plt.tight_layout()
    # plt.savefig('Results/Plots/'+ymlFile+'/'+str(expNum)+'.png', bbox_inches='tight')
    # plt.cla()
    # plt.close(fig)

    return absError,absErrorPcnt,jainActual,jainPred,jainsError,correation,corrConfidence,baseData

#%% add results to repository

def writeResults(expNum,method,p, s, probeBudget, sampleRate, seedSplit, al, absError,absErrorPcnt,jainActual,jainPred,jainsError,correation,corrConfidence,infTime,numSPQ,resultsFileName,baseData,ymlFile):

    resultRow = [expNum, method,p, s, probeBudget, sampleRate, seedSplit, al, absError,absErrorPcnt,jainActual,jainPred,jainsError,correation,corrConfidence,infTime,numSPQ]
    with open(resultsFileName,'a', newline='') as f:
        read_file = csv.writer(f)
        read_file.writerow(resultRow)

    baseData.to_csv('Results/Data/'+ymlFile+'/'+str(expNum)+'.csv')