#Import modules

import geopandas as gpd
import pandas as pd
from pathlib import Path
import numpy as np
from math import radians, cos, sin, asin, sqrt
import osmnx as ox
import networkx as nx

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

#%% Code parameters

p = Path(__file__).parents[2]
travel_speed = 4.5
outputDir = str(p) + '/Data/adjMx'

#%% Import Base Data

#Get west midlands shape files

wm_oas = gpd.read_file(str(p) + '/Data/west_midlands_OAs/west_midlands_OAs.shp')

#select coventry
wm_oas = wm_oas[wm_oas['LAD11CD'] == 'E08000026']

oa_info = pd.read_csv(str(p) + '/Data/oa_info.csv')

oa_info = oa_info.merge(wm_oas['OA11CD'], left_on = 'oa_id', right_on = 'OA11CD', how = 'inner')
oaIDIndex = list(oa_info['oa_id'])


#%% Import network from OSMNX for region

network_type = 'all'

max_lat_dwell = max(oa_info['oa_lat'].values)
min_lat_dwell = min(oa_info['oa_lat'].values)

max_lon_dwell = max(oa_info['oa_lon'].values)
min_lon_dwell = min(oa_info['oa_lon'].values)

max_lat = max_lat_dwell
min_lat = min_lat_dwell
max_lon = max_lon_dwell
min_lon = min_lon_dwell

height = max_lat - min_lat
width = max_lon - min_lon

buffer_height = height * 0.5
buffer_width = width * 0.5

north = max_lat + buffer_height
south = min_lat - buffer_height
west = min_lon - buffer_width
east = max_lon + buffer_width

#Pull Grid from Networkx

GWalk = ox.graph.graph_from_bbox(north,south,east,west,network_type = network_type, retain_all=True)

# add an edge attribute for time in minutes required to traverse each edge
meters_per_minute = travel_speed * 1000 / 60 #km per hour to m per minute
for u, v, data in GWalk.edges(data=True):
    data['time'] = data['length'] / meters_per_minute

GWalk = ox.project_graph(GWalk, to_crs = 'WGS84')

#%% Method 1 - Calculate Euclidean Distance

adjMx = np.zeros((len(oaIDIndex),len(oaIDIndex)))

counti = 0

for i in oaIDIndex:
    print(counti)
    baseOA = oa_info[oa_info['oa_id'] == i][['oa_lon','oa_lat']]
    countj = 0
    for j in oaIDIndex:
        targetOA = oa_info[oa_info['oa_id'] == j]
        adjMx[counti,countj] = haversine(baseOA['oa_lon'],baseOA['oa_lat'],targetOA['oa_lon'],targetOA['oa_lat'])
        countj += 1
    counti += 1


normalizeAndOutput(adjMx, 0.005, outputDir)

#%% Method 2 - Walking Distances

meters_per_minute = travel_speed * 1000 / 60

nearestNodes = []

#for each OA get nearest node in network
for oa in oaIDIndex:
    nextOA = (oa_info[oa_info['oa_id'] == oa]['oa_lat'].values[0],oa_info[oa_info['oa_id'] == oa]['oa_lon'].values[0])
    nearestNode = ox.get_nearest_node(GWalk, nextOA)
    nearestNodes.append(nearestNode)

#%%
oaToOaDist = np.zeros(shape=(len(oaIDIndex), len(oaIDIndex)))
shortestWalkingPaths = np.zeros(shape=(len(oaIDIndex), len(oaIDIndex)))
failPaths = []

count = 0
count_x = 0
for oa_x in nearestNodes:
    count_y = 0
    for oa_y in nearestNodes:
        count += 1
        if count % 100 == 0:
            percent_complete = (count / (len(nearestNodes) * len(nearestNodes))) * 100
            print(percent_complete)
        try:
            dist, path = nx.single_source_dijkstra(GWalk, source=oa_x, target=oa_y, weight='time')
            shortestWalkingPaths[count_x, count_y] = dist
        except:
            failPaths.append([oa_x,oa_y])
            shortestWalkingPaths[count_x, count_y] = None

        count_y += 1
    count_x += 1

#%% Method 3 - 1hop reachability

import shapely as sp

#%%
gtfs_data = str(p) + '/Data/tfwm_gtfs/'

calendar = pd.read_csv(gtfs_data+'calendar.txt', sep=",", header=0)
calendar_dates = pd.read_csv(gtfs_data+'calendar_dates.txt', sep=",", header=0)
routes = pd.read_csv(gtfs_data+'routes.txt', sep=",", header=0)
stop_times = pd.read_csv(gtfs_data+'stop_times.txt', sep=",", header=0, dtype={3:str,5:str})
stops = pd.read_csv(gtfs_data+'stops.txt', sep=",", header=0)
trips = pd.read_csv(gtfs_data+'trips.txt', sep=",", header=0)

#Extract geo points from stops data and convert into geo-point data type
stops_geo_points = []

for i,r in stops.iterrows():
    stops_geo_points.append(sp.geometry.Point(r['stop_lon'], r['stop_lat']))
stops['Geo Points'] = stops_geo_points

#%%

max_lon = wm_oas.total_bounds[2]
min_lon = wm_oas.total_bounds[0]
max_lat = wm_oas.total_bounds[3]
min_lat = wm_oas.total_bounds[1]

print('Max Lon ' + str(max_lon))
print('Min Lon ' + str(min_lon))
print('Max Lat ' + str(max_lat))
print('Min Lat ' + str(min_lat))

#%%

#Only select stops that are within the bounding box
within_bounding_box = []
for i,r in stops.iterrows():
    if r['stop_lon']>= min_lon and r['stop_lon'] <= max_lon and r['stop_lat']>= min_lat and r['stop_lat'] <= max_lat:
        within_bounding_box.append('Y')
    else:
        within_bounding_box.append('N')
stops['In WM Box'] = within_bounding_box

#%%

#Associate each bus stop to an OA

bus_stop_to_oa = []
count_stops = 0
for i_stop,r_stop in stops[stops['In WM Box'] == 'Y'].iterrows():
    count_stops += 1
    print('Next Stop : ' + str(count_stops))
    count_oas = 0
    oa_found = False
    for i_oa, r_oa in wm_oas.iterrows():
        count_oas += 1
        oa_shape = sp.geometry.asShape(r_oa['geometry'])
        if oa_shape.contains(r_stop['Geo Points']):
            bus_stop_to_oa.append(r_oa['OA11CD'])
            print('OA Found on Iteration : ' + str(count_oas))
            oa_found = True
    if oa_found == False:
        print('No OA found')
        bus_stop_to_oa.append(None)

wm_stops = stops[stops['In WM Box'] == 'Y']
wm_stops['OA Link'] = bus_stop_to_oa

wm_stops.to_csv(str(p) + '/Data/bus_network/wm_stops.csv')

#%% Method 4 - POI/OA Reachability



#%% Method 5 - OA to OA similarity

pois =     ['Hospital', 'Job Centre', 'Strategic Centre', 'School', 'Childcare', 'Rail Station']
stratums = ['Weekday (AM peak)','Sunday']

#%%

#Create matrix of |OAs| * |OAs| * (|P| * |S|)
#Index of |P| * |S| length

simMatrix = np.zeros((len(oaIDIndex),len(oaIDIndex),(len(pois) * len(stratums))))

#%%

#Get trips and calculated access cost
#Write SQL query to get trips

i = 0
j = 1
p = pois[0]
s = stratums[0]
oa1 = oaIDIndex[i]
oa2 = oaIDIndex[j]

#%%



#%%
sqlQuery = "select a.oa_id, a.trip_id, b.stratum from model_may2022.otp_trips as a left join model_may2022.trips as b on a.trip_id = b.trip_id where a.oa_id in {} and a.poi_id in (select distinct id from semantic.poi where type = '{}');".format(tuple(list(set(oaIndex))),p)

#%%
allPOItrips = database.execute_sql(sqlQuery, False, True)

#%%
#Calculate the peason correlation coefficient
#Compare

#Add to matrix



#%% Method 6 - Clustering based on OA / POI geo faetures

#For each poi and stratum



#Get basedata

#Put faetures into matrix

#%%
#Approach 1 - k-means binary association


#%%
#Approach 2 - hierarchical clustering, highly associate to those in cluster, then go up tree for next most associated



























