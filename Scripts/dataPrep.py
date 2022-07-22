#Import modules
import pandas as pd
import sqlite3
import geopandas as gpd
import numpy as np
from math import radians, cos, sin, asin, sqrt, atan2, pi

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


def direction_lookup(destination_y, destination_x, origin_y, origin_x):
    deltaX = destination_x - origin_x
    deltaY = destination_y - origin_y
    degrees_temp = atan2(deltaX, deltaY) / pi * 180
    if degrees_temp < 0:
        degrees_final = 360 + degrees_temp
    else:
        degrees_final = degrees_temp

    compass_brackets = ["N", "NE", "E", "SE", "S", "SW", "W", "NW", "N"]
    compass_lookup = round(degrees_final / 45)
    return compass_brackets[compass_lookup], degrees_final

#%% Load adjacency matrix

def loadAdj(aNp,oaIndex):

    #aNp = np.load(adjLoc)
    
    edgeList = []
    edgeWeightList = []
    
    for i in range(len(oaIndex)):
        for j in range(len(oaIndex)):
            if aNp[i,j] > 0:
                edgeList.append([i,j])
                edgeWeightList.append(aNp[i,j])
    
    edgeIndexNp = np.array(edgeList).T
    edgeWeightsNp = np.array(edgeWeightList)
    
    return edgeIndexNp,edgeWeightsNp

#%%

def getBaseTrainingData(shpFileLoc,oaInfoLoc,stratum,poiType,dbLoc,wm_oas,oa_info,oaIndex):
    
    #For the given stratum and poi type return a dataframe of all OAs with associated features
        
    #Get Results Summary
    sqlOAs = tuple(list(set(list(wm_oas['OA11CD']))))
    cnx = sqlite3.connect(dbLoc)
    resultsSummary = pd.read_sql_query("SELECT * FROM results_summary where poi_type = '{}' and stratum = '{}' and oa_id in {};".format(poiType,stratum,sqlOAs), cnx)
    
    #Associate OAs to POIs (gives to k nearest POIs per OA)
    oaToPoi = pd.read_sql_query("select distinct oa_id, cast(poi_id as integer) as poi_id from trips as a inner join poi as b on a.poi_id = b.id where a.oa_id in {} and b.type = '{}';".format(sqlOAs,poiType), cnx)
    
    poi = pd.read_sql_query("SELECT id as poi_id, snapped_longitude as poi_lon, snapped_latitude as poi_lat, type from poi where type = '{}';".format(poiType),cnx)
    
    #Get index of POIs
    poiInd = np.array(poi[poi['poi_id'].isin(list(set(list(oaToPoi['poi_id']))))]['poi_id'])
    poiLonLat = np.array(poi[poi['poi_id'].isin(list(set(list(oaToPoi['poi_id']))))][['poi_lon','poi_lat']])
    
    oaLonLat = np.array(oa_info[['oa_lon','oa_lat']])
    
    #Euclid Dist (avg, range)
    OaPoiDist = np.zeros((len(poiInd),len(oaIndex)))
    OaPoiDirection = np.zeros((len(poiInd),len(oaIndex)))
    
    #Direction
    #Get OA to POI dist
    for i in range(len(poiInd)):
        nextPOI = poiLonLat[i]
        for j in range(len(oaIndex)):
            nextOA = oaLonLat[j]
            OaPoiDist[i,j] = haversine(nextPOI[0], nextPOI[1], nextOA[0], nextOA[1])
            OaPoiDirection[i,j] = direction_lookup(nextPOI[0], nextPOI[1], nextOA[0], nextOA[1])[1]
    
    # Add average distance
    avgDist = []
    rangeDist = []
    
    avgDir = []
    rangeDir = []
    
    oaIndexes = []
    for i in list(resultsSummary['oa_id']):
        loc1 = []
        for j in list(oaToPoi[(oaToPoi['oa_id'] == i)]['poi_id']):
            loc1.append(list(poiInd).index(j))
        loc2 = oaIndex.index(i)
        
        avgDist.append(OaPoiDist[loc1,loc2].mean())
        rangeDist.append(OaPoiDist[loc1,loc2].max() - OaPoiDist[loc1,loc2].min())
        avgDir.append(OaPoiDirection[loc1,loc2].mean())
        rangeDir.append(OaPoiDirection[loc1,loc2].max() - OaPoiDirection[loc1,loc2].min())
        oaIndexes.append(loc2)
    
    resultsSummary['avgEuclidDist'] = avgDist
    resultsSummary['rangeEuclidDist'] = rangeDist
    resultsSummary['avgDirection'] = avgDir
    resultsSummary['rangeDirection'] = rangeDir
    
    #Append oa area
    resultsSummary = resultsSummary.merge(wm_oas[['OA11CD','Shape__Are']],left_on = 'oa_id',right_on = 'OA11CD')
    
    #Dist to nearest urban centre
    cov = (52.4078358796818, -1.5137840014354358)
    distToUrbanCentre = {}
    populationDensity = {}
    
    for i,r in oa_info.iterrows():   
        distToUrbanCentre[r['oa_id']] = haversine(r['oa_lon'], r['oa_lat'], cov[1], cov[0])
        populationDensity[r['oa_id']] = r['age_all_residents'] / wm_oas[wm_oas['OA11CD'] == r['oa_id']]['Shape__Are'].values[0]
    
    resultsSummary['distToUrbanCentre'] = resultsSummary['oa_id'].map(distToUrbanCentre)
    resultsSummary['populationDensity'] = resultsSummary['oa_id'].map(populationDensity)
    
    # Calculate average access cost
    resultsSummary['avgAccessCost'] = resultsSummary['sum_generalised_cost'] / resultsSummary['num_trips']
    
    #Get oa lat and lon
    resultsSummary = resultsSummary.merge(oa_info[['oa_id','oa_lat','oa_lon']], left_on = 'oa_id', right_on = 'oa_id', how = 'inner')

    return resultsSummary,oaIndex,poiLonLat, poiInd


#%%


