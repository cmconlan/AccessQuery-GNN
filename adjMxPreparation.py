#Import modules

import geopandas as gpd
import pandas as pd
from pathlib import Path
import numpy as np
from math import radians, cos, sin, asin, sqrt
#import osmnx as ox
import networkx as nx

from node2vec import Node2Vec as n2v

#%%

WINDOW = 1 # Node2Vec fit window
MIN_COUNT = 1 # Node2Vec min. count
BATCH_WORDS = 4 # Node2Vec batch words

#%% User functions

#Function to get euclidean distance
def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    # Radius of earth in kilometers is 6371
    km = 6371* c
    return km

def outputNumpy(file,array):
    with open(file, 'wb') as f:
        np.save(f, array)

def normalizeAndOutput(adjMx, k, outputDir):
    '''Normalize and input matrix and output as npy'''
    '''adjMx - n by n array'''
    '''k - parameter for gaussian kernel'''
    
    #Gaussian kernel normalization
    normalized_k = k
    adjMxFlat = adjMx[~np.isinf(adjMx)].flatten()
    std = adjMxFlat.std()

    adjMxNorm = np.exp(-np.square(adjMx / std))

    adjMxNorm[adjMxNorm < normalized_k] = 0
    adjMxNorm[adjMxNorm >= 1] = 0
    
    #Min max normalization
    
    adjMxNormMM = 1 - (adjMx - adjMx.min()) / (adjMx.max() - adjMx.min())
    
    #Output matrices
    
    outputNumpy(outputDir + '/adjMx_euclid_Gaus.npy', adjMxNorm)
    outputNumpy(outputDir + '/adjMx_euclid_MM.npy', adjMxNormMM)

#%%

shpFileLoc = 'Data/west_midlands_OAs/west_midlands_OAs.shp'
oaInfoLoc = 'Data/oa_info.csv'
travel_speed = 4.5

#listLAD11CD = ['E08000025','E08000026','E08000027','E08000028','E08000029','E08000030','E08000031']
listLAD11CD = ['E08000026','E08000025']

#%%
for region in listLAD11CD:

    print('REGION : ' + str(region))
    
    #Get west midlands shape files
    wm_oas = gpd.read_file(shpFileLoc)
    wm_oas = wm_oas[wm_oas['LAD11CD'] == region]
    
    oa_info = pd.read_csv(oaInfoLoc)
    oa_info = oa_info.merge(wm_oas[['OA11CD']], left_on = 'oa_id', right_on = 'OA11CD', how = 'inner')
    oaIndex = list(oa_info['oa_id'])
    
    #Calculate Euclidean Distance matrix
    
    euclidMx = np.zeros((len(oaIndex),len(oaIndex)))
    
    counti = 0
    
    for i in oaIndex:
        print(counti)
        baseOA = oa_info[oa_info['oa_id'] == i][['oa_lon','oa_lat']]
        countj = 0
        for j in oaIndex:
            targetOA = oa_info[oa_info['oa_id'] == j]
            euclidMx[counti,countj] = haversine(baseOA['oa_lon'],baseOA['oa_lat'],targetOA['oa_lon'],targetOA['oa_lat'])
            countj += 1
        counti += 1
    
    print('Adj Mx Constructed')
    
    # Output Matrix
    
    outputNumpy('Data/adjMx/' + str(region) + '/euclidMx.csv', euclidMx)
    
    # Use node2vec to calculate node embeddings
    
    # G = nx.DiGraph(euclidMx)
    
    # g_emb = n2v(
    #   G,
    #   dimensions=16,
    #   weight_key = 'weight'
    # )
    
    # mdl = g_emb.fit(
    #     vector_size = 16,
    #     window=WINDOW,
    #     min_count=MIN_COUNT,
    #     batch_words=BATCH_WORDS
    # )
    
    # print('NODE2VEC for euclid')
    
    # Ouput
    
    # outputNumpy('Data/adjMx/' + str(region) + '/euclidEmbeddings.csv', mdl.wv.vectors)
    
    #Normalize matrix
    
    normalized_k = 0.38
    adjMxFlat = euclidMx[~np.isinf(euclidMx)].flatten()
    std = adjMxFlat.std()
    
    adjMx = np.exp(-np.square(euclidMx / std))
    
    adjMx[adjMx < normalized_k] = 0
    adjMx[adjMx >= 1] = 0
    
    print('Adj Mx Gaus Formed')
    
    # Output
    outputNumpy('Data/adjMx/' + str(region) + '/adjMx.csv',adjMx)
    
    # Calculat embeddings on normalized matrix
    
    G = nx.DiGraph(adjMx)
    
    g_emb = n2v(
      G,
      dimensions=6,
      weight_key = 'weight'
    )
    
    mdl = g_emb.fit(
        vector_size = 6,
        window=WINDOW,
        min_count=MIN_COUNT,
        batch_words=BATCH_WORDS
    )
        
    print('NODE2VEC for gaus adj')
    
    # Output
    outputNumpy('Data/adjMx/' + str(region) + '/adjMxEmbeddings.csv', mdl.wv.vectors)