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

def performanceMetrics(predVector,inputData,testMask,y,shpFileLoc,oaInfoLoc,runRef,expNum):
    
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
    
    
    fig, axs = plt.subplots(3,2,figsize=(15,15))

    axs[0,0].hist(error, bins=20)
    axs[0,0].set_title('Histogram of errors (at OA level)')
    axs[0,0].set_xlabel('Error')

    axs[0,1].hist(errorPct, bins=20)
    axs[0,1].set_title('Histogram of error percent (at oa level)')
    axs[0,1].set_xlabel('Error Percent')

    inputData.plot.scatter(x = 'oa_lon', y = 'oa_lat', c = 'absError', colormap='viridis', ax = axs[1,0])
    axs[1,0].set_title('OAs plotted with error on colour scale')

    inputData.plot.scatter(x = 'oa_lon', y = 'oa_lat', c = 'absErrorPcnt', colormap='viridis', ax = axs[1,1])
    axs[1,1].set_title('OAs plotted with error percent on colour scale')

    wm_oas.plot(column='avgAccessCost', cmap='OrRd', scheme='quantiles', ax = axs[2,0])
    axs[2,0].set_title('Actual Access Costs')
    wm_oas.plot(column='predictedAccessCost', cmap='OrRd', scheme='quantiles', ax = axs[2,1])
    axs[2,1].set_title('Predicted Access Costs')

    plt.savefig('Results/Plots/'+runRef+'/'+str(expNum)+'.png', bbox_inches='tight')
    plt.cla()
    plt.close(fig)
    
    return absError,absErrorPcnt,jainActual,jainPred,jainsError,correation,corrConfidence

#%% add results to repository

def writeResults(expNum,method,p, s, probeBudget, sampleRate, seedSplit, aType, absError,absErrorPcnt,jainActual,jainPred,jainsError,correation,corrConfidence,infTime,numSPQ,resultsFileName):

    resultRow = [expNum, method,p, s, probeBudget, sampleRate, seedSplit, aType, absError,absErrorPcnt,jainActual,jainPred,jainsError,correation,corrConfidence, infTime,numSPQ]
    with open(resultsFileName,'a', newline='') as f:
        read_file = csv.writer(f)
        read_file.writerow(resultRow)
