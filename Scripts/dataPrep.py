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

#%% Create training data

# def baseTrainingData(dbLoc,shpFileLoc,oaInfoLoc,stratum,poiType):
    
#     #Download OTP results from database (DB currently July 2020 (?))
#     cnx = sqlite3.connect(dbLoc)
#     otpResults = pd.read_sql_query("SELECT * FROM otp_results_summary", cnx)
#     oa = pd.read_sql_query("SELECT * FROM oa", cnx)
#     poi = pd.read_sql_query("SELECT * FROM poi", cnx)

#     # Get Coventry OAs
#     wm_oas = gpd.read_file(shpFileLoc)
#     wm_oas = wm_oas[wm_oas['LAD11CD'] == 'E08000026']
#     oa_info = pd.read_csv(oaInfoLoc)
#     oa_info = oa_info.merge(wm_oas['OA11CD'], left_on = 'oa_id', right_on = 'OA11CD', how = 'inner')
#     oaIndex = list(oa_info['oa_id'])
    
#     #Filter on stratum and POI type
#     otpResultsFiltered = otpResults[otpResults['stratum'] == stratum]
#     otpResultsFiltered = otpResultsFiltered[otpResultsFiltered['oa_id'].isin(list(wm_oas['OA11CD']))]
#     otpResultsFiltered = otpResultsFiltered[otpResultsFiltered['poi_type'] == poiType]

#     # Link OAs to POIs
#     oaForSqlClause = tuple(list(set(list(wm_oas['OA11CD']))))
#     sqlQuery = 'select distinct oa_id, poi_id from otp_trips where oa_id in {};'.format(oaForSqlClause)
#     oa_to_poi = pd.read_sql_query(sqlQuery, cnx)

#     # Get POI type
#     oa_to_poi = oa_to_poi.merge(poi[['poi_id','type']],left_on = 'poi_id',right_on = 'poi_id')

#     #Get index of POIs
#     poiInd = np.array(poi[poi['poi_id'].isin(list(set(list(oa_to_poi['poi_id']))))]['poi_id'])
#     poiLonLat = np.array(poi[poi['poi_id'].isin(list(set(list(oa_to_poi['poi_id']))))][['poi_lon','poi_lat']])

#     #Get index of OAs
#     oaInd = np.array(oa[oa['oa_id'].isin(list(set(list(wm_oas['OA11CD']))))]['oa_id'])
#     oaLonLat = np.array(oa[oa['oa_id'].isin(list(set(list(wm_oas['OA11CD']))))][['oa_lon','oa_lat']])

#     OaPoiDist = np.zeros((len(poiInd),len(oaInd)))

#     #Get OA to POI dist
#     for i in range(len(poiInd)):
#         nextPOI = poiLonLat[i]
#         for j in range(len(oaInd)):
#             nextOA = oaLonLat[j]
#             OaPoiDist[i,j] = haversine(nextPOI[0], nextPOI[1], nextOA[0], nextOA[1])

#     # Add average distance
#     avgDist = []
#     oaIndexes = []
#     for i,r in otpResultsFiltered.iterrows():
#         loc1 = []
#         for j in list(oa_to_poi[(oa_to_poi['oa_id'] == r['oa_id']) & (oa_to_poi['type'] == r['poi_type'])]['poi_id']):
#             loc1.append(list(poiInd).index(j))
#         loc2 = list(oaInd).index(r['oa_id'])
        
#         avgDist.append(OaPoiDist[loc1,loc2].mean())
#         oaIndexes.append(loc2)

#     otpResultsFiltered['avgEuclidDist'] = avgDist

#     otpResultsFiltered['OAIndex'] = oaIndexes

#     # Calculate average access cost
#     otpResultsFiltered['avgAccessCost'] = otpResultsFiltered['sum_generalised_cost'] / otpResultsFiltered['num_trips']

#     # Append oa area
#     otpResultsFiltered = otpResultsFiltered.merge(wm_oas[['OA11CD','Shape__Are']],left_on = 'oa_id',right_on = 'OA11CD')

#     # Extra faetures

#     #Dist to nearest urban centre
#     cov = (52.4078358796818, -1.5137840014354358)

#     distToUrbanCentre = {}
#     populationDensity = {}

#     for i,r in oa_info.iterrows():   
#         distToUrbanCentre[r['oa_id']] = haversine(r['oa_lon'], r['oa_lat'], cov[1], cov[0])
#         populationDensity[r['oa_id']] = r['age_all_residents'] / wm_oas[wm_oas['OA11CD'] == r['oa_id']]['Shape__Are'].values[0]

#     otpResultsFiltered['distToUrbanCentre'] = otpResultsFiltered['oa_id'].map(distToUrbanCentre)
#     otpResultsFiltered['populationDensity'] = otpResultsFiltered['oa_id'].map(populationDensity)

#     return otpResultsFiltered, oaIndex, poiLonLat

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

# featureSets = {
#     'OA':['oa_id','Shape__Are','Count Bus Stops in OA','Stop IDs in OA','OA Level Features'],
#     '200':['oa_id','Shape__Are','Count Bus Stops within 200m', 'Stop IDs within 200m','200m Level Features'],
#     '500':['oa_id','Shape__Are','Count Bus Stops within 500m','Stop IDs within 500m', '500m Level Features'],
#     '1000':['oa_id','Shape__Are','Count Bus Stops within 1km', 'Stop IDs within 1km','1km Level Features'],
#     }


# poiType = 'School'
# stratum = 'Weekday (AM peak)'
# shpFileLoc = 'Data/west_midlands_OAs/west_midlands_OAs.shp'
# oaInfoLoc = 'Data/oa_info.csv'

#%%
# def baseTrainingData(shpFileLoc,oaInfoLoc,stratum,poiType,database,featureSets,featureGrain):
# # def baseTrainingData(shpFileLoc,oaInfoLoc,stratum,poiType,database,featureSets,featureGrain,features):    

#     otpResults = database.execute_sql('SELECT * FROM results_may2022.results_summary', False, True)
#     oa = database.execute_sql('SELECT oa11 as oa_id, snapped_longitude as oa_lon, snapped_latitude as oa_lat FROM semantic.oa', False, True)
#     poi = database.execute_sql("SELECT id as poi_id, snapped_longitude as poi_lon, snapped_latitude as poi_lat, type FROM semantic.poi where type = '{}';".format(poiType), False, True)
    
#     # Get Coventry OAs
#     wm_oas = gpd.read_file(shpFileLoc)
#     wm_oas = wm_oas[wm_oas['LAD11CD'] == 'E08000026']
#     oa_info = pd.read_csv(oaInfoLoc)
#     oa_info = oa_info.merge(wm_oas[['OA11CD']], left_on = 'oa_id', right_on = 'OA11CD', how = 'inner')
#     oaIndex = list(oa_info['oa_id'])
#     OALvlFeatures = pd.read_csv('Data/bus_network/OALevelFeatures.csv')
    
#     #Filter on stratum and POI type
#     otpResultsFiltered = otpResults[otpResults['stratum'] == stratum]
#     otpResultsFiltered = otpResultsFiltered[otpResultsFiltered['oa_id'].isin(list(wm_oas['OA11CD']))]
#     otpResultsFiltered = otpResultsFiltered[otpResultsFiltered['poi_type'] == poiType]
    
#     # Link OAs to POIs
#     oaForSqlClause = tuple(list(set(list(wm_oas['OA11CD']))))
#     sqlQuery = 'select distinct oa_id, poi_id from model_may2022.otp_trips where oa_id in {};'.format(oaForSqlClause)
#     oa_to_poi = database.execute_sql(sqlQuery, False, True)
    
#     # Get POI type
#     oa_to_poi = oa_to_poi.merge(poi[['poi_id','type']],left_on = 'poi_id',right_on = 'poi_id', how = 'inner')
    
    
#     #Get index of POIs
#     poiInd = np.array(poi[poi['poi_id'].isin(list(set(list(oa_to_poi['poi_id']))))]['poi_id'])
#     poiLonLat = np.array(poi[poi['poi_id'].isin(list(set(list(oa_to_poi['poi_id']))))][['poi_lon','poi_lat']])
    
#     #Get index of OAs
#     oaInd = np.array(oa[oa['oa_id'].isin(list(set(list(wm_oas['OA11CD']))))]['oa_id'])
#     oaLonLat = np.array(oa[oa['oa_id'].isin(list(set(list(wm_oas['OA11CD']))))][['oa_lon','oa_lat']])
    
#     # Dynamic Feature Selection
    
#     #Euclid Dist (avg, range)
#     OaPoiDist = np.zeros((len(poiInd),len(oaInd)))
#     OaPoiDirection = np.zeros((len(poiInd),len(oaInd)))
    
#     #Direction
#     #Get OA to POI dist
#     for i in range(len(poiInd)):
#         nextPOI = poiLonLat[i]
#         for j in range(len(oaInd)):
#             nextOA = oaLonLat[j]
#             OaPoiDist[i,j] = haversine(nextPOI[0], nextPOI[1], nextOA[0], nextOA[1])
#             OaPoiDirection[i,j] = direction_lookup(nextPOI[0], nextPOI[1], nextOA[0], nextOA[1])[1]
    
#     # Add average distance
#     avgDist = []
#     rangeDist = []
    
#     avgDir = []
#     rangeDir = []
    
#     oaIndexes = []
#     for i,r in otpResultsFiltered.iterrows():
#         loc1 = []
#         for j in list(oa_to_poi[(oa_to_poi['oa_id'] == r['oa_id']) & (oa_to_poi['type'] == r['poi_type'])]['poi_id']):
#             loc1.append(list(poiInd).index(j))
#         loc2 = list(oaInd).index(r['oa_id'])
        
#         avgDist.append(OaPoiDist[loc1,loc2].mean())
#         rangeDist.append(OaPoiDist[loc1,loc2].max() - OaPoiDist[loc1,loc2].min())
#         avgDir.append(OaPoiDirection[loc1,loc2].mean())
#         rangeDir.append(OaPoiDirection[loc1,loc2].max() - OaPoiDirection[loc1,loc2].min())
#         oaIndexes.append(loc2)
    
#     otpResultsFiltered['avgEuclidDist'] = avgDist
#     otpResultsFiltered['rangeEuclidDist'] = rangeDist
#     otpResultsFiltered['avgDirection'] = avgDir
#     otpResultsFiltered['rangeDirection'] = rangeDir
        
#     otpResultsFiltered['stopDensity'] = 0
#     otpResultsFiltered['routesPerStop'] = 0
#     otpResultsFiltered['bussesPerRoute'] = 0
    
#     #Append oa area
#     otpResultsFiltered = otpResultsFiltered.merge(wm_oas[['OA11CD','Shape__Are']],left_on = 'oa_id',right_on = 'OA11CD')
    
#     #Dist to nearest urban centre
#     cov = (52.4078358796818, -1.5137840014354358)
#     distToUrbanCentre = {}
#     populationDensity = {}

#     for i,r in oa_info.iterrows():   
#         distToUrbanCentre[r['oa_id']] = haversine(r['oa_lon'], r['oa_lat'], cov[1], cov[0])
#         populationDensity[r['oa_id']] = r['age_all_residents'] / wm_oas[wm_oas['OA11CD'] == r['oa_id']]['Shape__Are'].values[0]

#     otpResultsFiltered['distToUrbanCentre'] = otpResultsFiltered['oa_id'].map(distToUrbanCentre)
#     otpResultsFiltered['populationDensity'] = otpResultsFiltered['oa_id'].map(populationDensity)
    
#     # Calculate average access cost
#     otpResultsFiltered['avgAccessCost'] = otpResultsFiltered['sum_generalised_cost'] / otpResultsFiltered['num_trips']
    
#     for i,r in OALvlFeatures[featureSets[featureGrain]].iterrows():
#         if isinstance(r[featureSets[featureGrain][4]], str):
#             if featureGrain == 'OA':
#                 areaSize = r[featureSets[featureGrain][1]]
#             else:
#                 areaSize = pi * (int(featureGrain) ** 2)
#             featuresI = r[featureSets[featureGrain][4]][1:-1].split(",")
#             #Add stop density
#             otpResultsFiltered.loc[otpResultsFiltered[otpResultsFiltered['oa_id'] == r[featureSets[featureGrain][0]]].index,'stopDensity'] = int(featuresI[0]) / areaSize
#             #Add Routes per Stop
#             otpResultsFiltered.loc[otpResultsFiltered[otpResultsFiltered['oa_id'] == r[featureSets[featureGrain][0]]].index,'routesPerStop'] = int(featuresI[1]) / int(featuresI[0])
#             #Add busses per route
#             otpResultsFiltered.loc[otpResultsFiltered[otpResultsFiltered['oa_id'] == r[featureSets[featureGrain][0]]].index,'bussesPerRoute'] = int(featuresI[2]) / int(featuresI[1])
            
#             #Add lat/lon
            
#     otpResultsFiltered = otpResultsFiltered.merge(oa_info[['oa_id','oa_lat','oa_lon']], left_on = 'oa_id', right_on = 'oa_id', how = 'inner')
            
#     # oaFeatureVec = otpResultsFiltered.copy()

#     # for i,r in OALvlFeatures[featureSets['OA']].iterrows():
#     #     if isinstance(r[featureSets['OA'][4]], str):
            
#     #         areaSize = r[featureSets['OA'][1]]

#     #         featuresI = r[featureSets['OA'][4]][1:-1].split(",")
#     #         #Add stop density
#     #         oaFeatureVec.loc[oaFeatureVec[oaFeatureVec['oa_id'] == r[featureSets['OA'][0]]].index,'stopDensity'] = int(featuresI[0]) / areaSize
#     #         #Add Routes per Stop
#     #         oaFeatureVec.loc[oaFeatureVec[oaFeatureVec['oa_id'] == r[featureSets['OA'][0]]].index,'routesPerStop'] = int(featuresI[1]) / int(featuresI[0])
#     #         #Add busses per route
#     #         oaFeatureVec.loc[oaFeatureVec[oaFeatureVec['oa_id'] == r[featureSets['OA'][0]]].index,'bussesPerRoute'] = int(featuresI[2]) / int(featuresI[1])


#     # oaFeatureVec = oaFeatureVec[features].to_numpy()

#     return otpResultsFiltered, oaIndex, poiLonLat
    # return otpResultsFiltered, oaIndex, poiLonLat, oaFeatureVec


#%%
def getBaseTrainingData(shpFileLoc,oaInfoLoc,stratum,poiType,dbLoc):
    
    #For the given stratum and poi type return a dataframe of all OAs with associated features
    
    # Read in base data - shapefile / OA info / OA index
    wm_oas = gpd.read_file(shpFileLoc)
    wm_oas = wm_oas[wm_oas['LAD11CD'] == 'E08000026']
    oa_info = pd.read_csv(oaInfoLoc)
    oa_info = oa_info.merge(wm_oas[['OA11CD']], left_on = 'oa_id', right_on = 'OA11CD', how = 'inner')
    oaIndex = list(oa_info['oa_id'])
    
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


