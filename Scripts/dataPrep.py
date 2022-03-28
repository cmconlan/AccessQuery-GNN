#Import modules
import pandas as pd
import sqlite3
import geopandas as gpd
import numpy as np
from math import radians, cos, sin, asin, sqrt

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

#%% Create training data

def baseTrainingData(dbLoc,shpFileLoc,oaInfoLoc,stratum,poiType):
    
    #Download OTP results from database (DB currently July 2020 (?))
    cnx = sqlite3.connect(dbLoc)
    otpResults = pd.read_sql_query("SELECT * FROM otp_results_summary", cnx)
    oa = pd.read_sql_query("SELECT * FROM oa", cnx)
    poi = pd.read_sql_query("SELECT * FROM poi", cnx)

    # Get Coventry OAs
    wm_oas = gpd.read_file(shpFileLoc)
    wm_oas = wm_oas[wm_oas['LAD11CD'] == 'E08000026']
    oa_info = pd.read_csv(oaInfoLoc)
    oa_info = oa_info.merge(wm_oas['OA11CD'], left_on = 'oa_id', right_on = 'OA11CD', how = 'inner')
    oaIndex = list(oa_info['oa_id'])
    
    #Filter on stratum and POI type
    otpResultsFiltered = otpResults[otpResults['stratum'] == stratum]
    otpResultsFiltered = otpResultsFiltered[otpResultsFiltered['oa_id'].isin(list(wm_oas['OA11CD']))]
    otpResultsFiltered = otpResultsFiltered[otpResultsFiltered['poi_type'] == poiType]

    # Link OAs to POIs
    oaForSqlClause = tuple(list(set(list(wm_oas['OA11CD']))))
    sqlQuery = 'select distinct oa_id, poi_id from otp_trips where oa_id in {};'.format(oaForSqlClause)
    oa_to_poi = pd.read_sql_query(sqlQuery, cnx)

    # Get POI type
    oa_to_poi = oa_to_poi.merge(poi[['poi_id','type']],left_on = 'poi_id',right_on = 'poi_id')

    #Get index of POIs
    poiInd = np.array(poi[poi['poi_id'].isin(list(set(list(oa_to_poi['poi_id']))))]['poi_id'])
    poiLonLat = np.array(poi[poi['poi_id'].isin(list(set(list(oa_to_poi['poi_id']))))][['poi_lon','poi_lat']])

    #Get index of OAs
    oaInd = np.array(oa[oa['oa_id'].isin(list(set(list(wm_oas['OA11CD']))))]['oa_id'])
    oaLonLat = np.array(oa[oa['oa_id'].isin(list(set(list(wm_oas['OA11CD']))))][['oa_lon','oa_lat']])

    OaPoiDist = np.zeros((len(poiInd),len(oaInd)))

    #Get OA to POI dist
    for i in range(len(poiInd)):
        nextPOI = poiLonLat[i]
        for j in range(len(oaInd)):
            nextOA = oaLonLat[j]
            OaPoiDist[i,j] = haversine(nextPOI[0], nextPOI[1], nextOA[0], nextOA[1])

    # Add average distance
    avgDist = []
    oaIndexes = []
    for i,r in otpResultsFiltered.iterrows():
        loc1 = []
        for j in list(oa_to_poi[(oa_to_poi['oa_id'] == r['oa_id']) & (oa_to_poi['type'] == r['poi_type'])]['poi_id']):
            loc1.append(list(poiInd).index(j))
        loc2 = list(oaInd).index(r['oa_id'])
        
        avgDist.append(OaPoiDist[loc1,loc2].mean())
        oaIndexes.append(loc2)

    otpResultsFiltered['avgEuclidDist'] = avgDist

    otpResultsFiltered['OAIndex'] = oaIndexes

    # Calculate average access cost
    otpResultsFiltered['avgAccessCost'] = otpResultsFiltered['sum_generalised_cost'] / otpResultsFiltered['num_trips']

    # Append oa area
    otpResultsFiltered = otpResultsFiltered.merge(wm_oas[['OA11CD','Shape__Are']],left_on = 'oa_id',right_on = 'OA11CD')

    # Extra faetures

    #Dist to nearest urban centre
    cov = (52.4078358796818, -1.5137840014354358)

    distToUrbanCentre = {}
    populationDensity = {}

    for i,r in oa_info.iterrows():   
        distToUrbanCentre[r['oa_id']] = haversine(r['oa_lon'], r['oa_lat'], cov[1], cov[0])
        populationDensity[r['oa_id']] = r['age_all_residents'] / wm_oas[wm_oas['OA11CD'] == r['oa_id']]['Shape__Are'].values[0]

    otpResultsFiltered['distToUrbanCentre'] = otpResultsFiltered['oa_id'].map(distToUrbanCentre)
    otpResultsFiltered['populationDensity'] = otpResultsFiltered['oa_id'].map(populationDensity)

    return otpResultsFiltered, oaIndex

#%% Load adjacency matrix

def loadAdj(adjLoc,oaIndex):

    aNp = np.load(adjLoc)
    
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

#p = 'School'
#s = 'Weekday (AM peak)'
#dbloc = 'C:/Users/chris/Documents/University/Warwick/Research/Transport Access/Current Server DB//tfwm.db'
#shpLoc = 'C:/Users/chris/My Drive/University/Working Folder/Transport Access Tool/ML Supported Bus Network/Data/west_midlands_OAs/west_midlands_OAs.shp'
#baseTraining,oaIndex = baseTrainingData(dbloc,shpLoc,s,p)

#adjLoc = 'C:/Users/chris/My Drive/University/Working Folder/Transport Access Tool/AccessWithGNN/AccessWithGNN/data/adjMx_1hop_Gaus.npy'
#edgeIndexNp,edgeWeightsNp = loadAdj(adjLoc,oaIndex)