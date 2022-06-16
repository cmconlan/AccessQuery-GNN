from Scripts.dataPrep import baseTrainingData
from Scripts.util.database import Database
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
import geopandas as gpd
import pandas as pd
from pathlib import Path
from math import radians, cos, sin, asin, sqrt

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

#%% Set Up

p = Path(__file__).parents[2]
travel_speed = 4.5
outputDir = str(p) + '/Data/adjMx'

#Get west midlands shape files

wm_oas = gpd.read_file(str(p) + '/Data/west_midlands_OAs/west_midlands_OAs.shp')

#select coventry
wm_oas = wm_oas[wm_oas['LAD11CD'] == 'E08000026']
oa_info = pd.read_csv(str(p) + '/Data/oa_info.csv')
oa_info = oa_info.merge(wm_oas['OA11CD'], left_on = 'oa_id', right_on = 'OA11CD', how = 'inner')
oaIDIndex = list(oa_info['oa_id'])

shpLoc = 'Data/west_midlands_OAs/west_midlands_OAs.shp'
oaInfoLoc = 'Data/oa_info.csv'

database = Database.get_instance()

featureSets = {
    'OA':['oa_id','Shape__Are','Count Bus Stops in OA','Stop IDs in OA','OA Level Features'],
    '200':['oa_id','Shape__Are','Count Bus Stops within 200m', 'Stop IDs within 200m','200m Level Features'],
    '500':['oa_id','Shape__Are','Count Bus Stops within 500m','Stop IDs within 500m', '500m Level Features'],
    '1000':['oa_id','Shape__Are','Count Bus Stops within 1km', 'Stop IDs within 1km','1km Level Features'],
    }

#Align POIs and Stratums with experiments
pois =     ['Hospital', 'Job Centre', 'Strategic Centre', 'School', 'Childcare', 'Rail Station']
stratums = ['Weekday (AM peak)','Sunday']
#Keep grain at OA level
g = 'OA'

#%% Approach 1/2 - Euclidean Distance Matrix (Min/Max and Gaussian Normalization)

euclidMx = np.zeros((len(oaIDIndex),len(oaIDIndex)))

counti = 0

for i in oaIDIndex:
    print(counti)
    baseOA = oa_info[oa_info['oa_id'] == i][['oa_lon','oa_lat']]
    countj = 0
    for j in oaIDIndex:
        targetOA = oa_info[oa_info['oa_id'] == j]
        euclidMx[counti,countj] = haversine(baseOA['oa_lon'],baseOA['oa_lat'],targetOA['oa_lon'],targetOA['oa_lat'])
        countj += 1
    counti += 1

#%%

with open('C:/Users/chris/My Drive/University/Working Folder/Transport Access Tool/AccessQuery-GNN/Data/bus_network/euclidMatrix.csv', 'wb') as f:
    np.save(f, euclidMx)

#%%
euclidMx.to_csv('C:/Users/chris/My Drive/University/Working Folder/Transport Access Tool/AccessQuery-GNN/Data\bus_network/euclidMatrix.csv')

#%%
#Get basedata
p = pois[0]
s = stratums[0]

baseData, oaIndex, poiLonLat = baseTrainingData(shpLoc,oaInfoLoc,s,p,database,featureSets,g)

#Put faetures into matrix
featuresNP = baseData[['Shape__Are','avgEuclidDist','rangeEuclidDist','avgDirection','rangeDirection','stopDensity','routesPerStop','bussesPerRoute']].to_numpy()


#%%
#Approach 2 - k-means binary association small groups


scaler = StandardScaler()
scaled_features = scaler.fit_transform(featuresNP)

nClusters = int(scaled_features.shape[0]/20)

kmeans = KMeans(init="random",n_clusters=nClusters,n_init=10,max_iter=1000,random_state=42)
kmeans.fit(scaled_features)

clusterLabels = kmeans.labels_

#Construct graph

#Get OA index
#Create blank matrix
#Cycle through index, for each oa test each other OA. If label same 1, else 0

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

#%% Approach 2 - k-mean larger groups, then use eucliden distance to normalise within each group

nClusters = int(scaled_features.shape[0]/40)

kmeans = KMeans(init="random",n_clusters=nClusters,n_init=10,max_iter=1000,random_state=42)
kmeans.fit(scaled_features)

clusterLabels = kmeans.labels_

#%% Normalise within clusters by Euclidean distance

#Import euclidean distance matrix
adjMx = np.zeros((len(oaIndex),len(oaIndex)))

counti = 0

for i in oaIndex:
    print(counti)
    baseOA = oa_info[oa_info['oa_id'] == i][['oa_lon','oa_lat']]
    countj = 0
    for j in oaIDIndex:
        targetOA = oa_info[oa_info['oa_id'] == j]
        adjMx[counti,countj] = haversine(baseOA['oa_lon'],baseOA['oa_lat'],targetOA['oa_lon'],targetOA['oa_lat'])
        countj += 1
    counti += 1
    
#%%

#For each OA
oa = oaIndex[0]
#Get index of OAs in same cluster
oaCluster = clusterLabels[0]

#%%

clusterIndex = np.where(clusterLabels == oaCluster)[0]
oaDistances = euclidMx[0,clusterIndex]

#Normalize by Eucldean distance (1 -)

adjMxNormMM = 1 - (oaDistances - oaDistances.min()) / (oaDistances.max() - oaDistances.min())

#Add to adj matrix

adjMx = np.zeros((len(oaIndex),len(oaIndex)))

countI = 0
for i in oaIndex:
    oaCluster = clusterLabels[countI] 
    clusterIndex = np.where(clusterLabels == oaCluster)[0]
    oaDistances = euclidMx[countI,clusterIndex]
    adjMxNormMM = 1 - (oaDistances - oaDistances.min()) / (oaDistances.max() - oaDistances.min())
    countJ = 0
    for j in clusterIndex:
        adjMx[countI,j] = adjMxNormMM[countJ]
        countJ += 1
    countI += 1

#%%
#Approach 3 - hierarchical clustering, highly associate to those in cluster, then go up tree for next most associated



#%%


























