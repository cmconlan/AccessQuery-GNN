import shapely as sp
from pathlib import Path
import pandas as pd
import geopandas as gpd
import json
import networkx as nx
import osmnx as ox
import statistics
import datetime
import numpy as np
from math import radians, cos, sin, asin, sqrt

#%%

def getMeanTime(times):
    timeInSeconds = []

    for i in times:
        timeInSeconds.append(3600 * i.hour + 60 * i.minute + i.second)

    meanSeconds = statistics.mean(timeInSeconds)

    hour = int(meanSeconds / 3600)
    minute = int((meanSeconds % 3600) / 60)
    second = int(meanSeconds % 60)

    return datetime.time(int(hour), int(minute), int(second))

def getTravelTime(start, end):

    startInSeconds = 3600 * start.hour + 60 * start.minute + start.second
    endInSeconds = 3600 * end.hour + 60 * end.minute + end.second
    diffSeconds = endInSeconds - startInSeconds
    return diffSeconds

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

#%%

p = Path(__file__).parents[2]
travel_speed = 4.5
outputDir = str(p) + '/Data/adjMx'

wm_oas = gpd.read_file(str(p) + '/Data/west_midlands_OAs/west_midlands_OAs.shp')

#select coventry
wm_oas = wm_oas[wm_oas['LAD11CD'] == 'E08000026']


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

#Get the parameters for the bounding box (e.g., north, south, east, west)
max_lon = wm_oas.total_bounds[2]
min_lon = wm_oas.total_bounds[0]
max_lat = wm_oas.total_bounds[3]
min_lat = wm_oas.total_bounds[1]

print('Max Lon ' + str(max_lon))
print('Min Lon ' + str(min_lon))
print('Max Lat ' + str(max_lat))
print('Min Lat ' + str(min_lat))

#Only select stops that are within the bounding box
within_bounding_box = []
for i,r in stops.iterrows():
    if r['stop_lon']>= min_lon and r['stop_lon'] <= max_lon and r['stop_lat']>= min_lat and r['stop_lat'] <= max_lat:
        within_bounding_box.append('Y')
    else:
        within_bounding_box.append('N')
stops['In WM Box'] = within_bounding_box

#%%
#------------------ PART 1 - ASSOCIATE EACH BUS STOP TO AN OA ---------------------------------


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

#%% 
#------------------ PART 2 - TRANSLATE ROUTES TO EDGES ON NETWORK ---------------------------------


route_count = 0
len_routes = len(list(routes['route_id']))
trip_id_short = 0
#trip_details = pd.DataFrame(columns = ['stop_id', 'OA Link', 'stop_sequence', 'departure_time','trip_id','route_id'])
trip_details = pd.DataFrame(columns = ['stop_id', 'OA Link', 'stop_sequence', 'departure_time','trip_id','route_id'])
tripID_dict = {}

#For each route
for route_id in list(routes['route_id']):
    route_count += 1
    print('Next Route')
    print('Route : ' + str(route_count) + " of : " + str(len_routes))
    route = trips[trips['route_id'] == route_id]
    print('Trips in route : ' + str(route.shape[0]))
    trip_count = 0
    #Get each unique trip per route
    for trip_id in list(route['trip_id']):
        trip_count += 1
        trip_id_short += 1
        if trip_count % 10 == 0:
            print('Trips Processed : ' + str(trip_count))
        trip = stop_times[stop_times['trip_id'] == trip_id]
        oa_edges_to_add = trip.merge(wm_stops[['stop_id', 'OA Link']], left_on='stop_id', right_on='stop_id',how='inner')
        data_to_add = oa_edges_to_add[['stop_id', 'OA Link', 'stop_sequence', 'departure_time']]
        data_to_add['trip_id'] = trip_id_short
        data_to_add['route_id'] = route_id
        trip_details = trip_details.append(data_to_add)
        tripID_dict[trip_id_short] = trip_id

trip_details = pd.DataFrame(trip_details)

trip_details.to_csv(str(p) + '/Data/bus_network/trip_details.csv')

with open(str(p) + '/Data/bus_network/trip_id_dict.txt', 'w') as file:
    file.write(json.dumps(tripID_dict))  # use `json.loads` to do the reverse
  

#%%
#------------------ PART 3 - CALCULATE EDGES TO ADD TO NETWORK ---------------------------------

oa_info = pd.read_csv(str(p) + '/Data/oa_info.csv')
oa_info = oa_info.merge(wm_oas['OA11CD'], left_on = 'oa_id', right_on = 'OA11CD', how = 'inner')

wm_stops = pd.read_csv(str(p) + '/Data/bus_network/wm_stops.csv')

trip_details = pd.read_csv(str(p) + '/Data/bus_network/trip_details.csv')
trip_details = trip_details.drop(trip_details.columns[0], axis=1)

trip_details_old = pd.read_csv('C:/Users/chris/My Drive/University/Working Folder/Transport Access Tool/ML Supported Bus Network/Data/trip_details.csv')

with open(str(p) + '/Data/bus_network/trip_id_dict.txt') as json_file:
    trip_id_dict = json.load(json_file)


nodes_with_features = []

for i,r in oa_info.iterrows():
    feature_dict = {}
    feature_dict['centroid'] = [r['oa_lon'],r['oa_lat']]
    feature_dict['linked_bus_stops'] = list(wm_stops[wm_stops['OA Link'] == r['oa_id']]['stop_id'])
    nodes_with_features.append(tuple([r['oa_id'],feature_dict]))


G = nx.DiGraph()
G.add_nodes_from(nodes_with_features)

edges_to_add = dict()
count_broken_trips = 0

numRecords = len(list(trip_details['trip_id'].value_counts().index))
count = 0


for id in list(trip_details['trip_id'].value_counts().index):
    count += 1
    if count % numRecords == 0:
        print('Progress : ' + str(count / numRecords))
    tripIDOrig = trip_id_dict[str(id)]
    tripDays = calendar[calendar['service_id'] == trips[trips['trip_id'] == tripIDOrig]['service_id'].values[0]]
    print(id)
    if tripDays.shape[0] > 0:

        if tripDays['tuesday'].values[0] == 1:

            nextTrip = trip_details[trip_details['trip_id'] == id]
            nextTrip = nextTrip.sort_values(by='stop_sequence')
            firstGroup = True
            lenTrip = len(nextTrip)
            lastOA = None
            visitedOAs = []
            obs = 0

            for i, r in nextTrip.iterrows():
                obs += 1
                # print('---------------------')
                # print('NEXT ITERATION')
                # print('obs : ' + str(obs))
                # Get Current OA

                #Skip any journeys that carry over to next day (for simplicity)
                if r['departure_time'][0:2] == '24':
                    # print('Into next day - skip rest of trip')
                    count_broken_trips += 1
                    break

                currentOA = r['OA Link']

                if lastOA == None:
                    # print('First It')
                    oaTimes = []
                    # oaTimes.append(r['departure_time'])

                else:
                    # Same OA
                    if currentOA == lastOA:
                        pass

                    # If not same OA - new grouping identified
                    else:
                        # print('Grouping Identified :  ' + str(lastOA))

                        # if not first OA, add edge, otherwise reset data types and move on

                        if firstGroup:
                            # print('First Grouping Identified')
                            depTimes = oaTimes
                            uNode = lastOA
                            firstGroup = False
                            oaTimes = []
                        else:
                            arrTimes = oaTimes
                            vNode = lastOA

                            edge = tuple([uNode, vNode])
                            # print(edge)
                            # print('Dep Times : ' + str(depTimes))
                            # print('Arr Times : ' + str(arrTimes))
                            # print('ID : ' + str(id))
                            if vNode not in visitedOAs:
                                # print('ADD THIS NODE')
                                # check if key exists in dict
                                if edge not in edges_to_add:
                                    edges_to_add[edge] = {
                                        'depTimes': [],
                                        'arrTimes': [],
                                        'tripIDs': [],
                                        'routeIDs': [],
                                        'travelTime': []
                                    }
                                    #edges_to_add[edge]['depTimes'].append(depTimes)
                                    #edges_to_add[edge]['arrTimes'].append(arrTimes)
                                    edges_to_add[edge]['depTimes'].append(getMeanTime(depTimes))
                                    edges_to_add[edge]['arrTimes'].append(getMeanTime(arrTimes))
                                    edges_to_add[edge]['tripIDs'].append(id)
                                    edges_to_add[edge]['routeIDs'].append(r['route_id'])
                                    edges_to_add[edge]['travelTime'].append(getTravelTime(getMeanTime(depTimes), getMeanTime(arrTimes)))

                                else:
                                    edges_to_add[edge]['depTimes'].append(getMeanTime(depTimes))
                                    edges_to_add[edge]['arrTimes'].append(getMeanTime(arrTimes))
                                    edges_to_add[edge]['tripIDs'].append(id)
                                    edges_to_add[edge]['routeIDs'].append(r['route_id'])
                                    edges_to_add[edge]['travelTime'].append(getTravelTime(getMeanTime(depTimes), getMeanTime(arrTimes)))

                                visitedOAs.append(vNode)

                            depTimes = arrTimes
                            uNode = lastOA
                            oaTimes = []

                if obs == lenTrip:
                    # print('End of trip')
                    arrTimes = oaTimes

                    if len(arrTimes) > 0:
                        vNode = lastOA

                        edge = tuple([uNode, vNode])
                        print(edge)
                        # print('Dep Times : ' + str(depTimes))
                        # print('Arr Times : ' + str(arrTimes))
                        # print('ID : ' + str(id))
                        if vNode not in visitedOAs:
                            # print('ADD THIS NODE')
                            # check if key exists in dict
                            if edge not in edges_to_add:
                                edges_to_add[edge] = {
                                    'depTimes': [],
                                    'arrTimes': [],
                                    'tripIDs': [],
                                    'routeIDs': [],
                                    'travelTime': []
                                }
                                edges_to_add[edge]['depTimes'].append(getMeanTime(depTimes))
                                edges_to_add[edge]['arrTimes'].append(getMeanTime(arrTimes))
                                edges_to_add[edge]['tripIDs'].append(id)
                                edges_to_add[edge]['routeIDs'].append(r['route_id'])
                                edges_to_add[edge]['travelTime'].append(getTravelTime(getMeanTime(depTimes), getMeanTime(arrTimes)))
                            else:
                                edges_to_add[edge]['depTimes'].append(getMeanTime(depTimes))
                                edges_to_add[edge]['arrTimes'].append(getMeanTime(arrTimes))
                                edges_to_add[edge]['tripIDs'].append(id)
                                edges_to_add[edge]['routeIDs'].append(r['route_id'])
                                edges_to_add[edge]['travelTime'].append(getTravelTime(getMeanTime(depTimes), getMeanTime(arrTimes)))
                            visitedOAs.append(vNode)

                #oaTimes.append(r['departure_time'])
                oaTimes.append(datetime.datetime.strptime(r['departure_time'], '%H:%M:%S').time())
                lastOA = currentOA
        else:
            pass

success_count = 0
fail_count = 0

for key, value in edges_to_add.items():
    if key[0] in list(oa_info['oa_id']) and key[1] in list(oa_info['oa_id']):

        edgeTimes = pd.DataFrame(columns=['depTimes', 'arrTimes', 'tripIDs', 'routeIDs', 'travelTime'])
        edgeTimes['depTimes'] = value['depTimes']
        edgeTimes['arrTimes'] = value['arrTimes']
        edgeTimes['tripIDs'] = value['tripIDs']
        edgeTimes['routeIDs'] = value['routeIDs']
        edgeTimes['travelTime'] = value['travelTime']

        # set index as departure time
        edgeTimes = edgeTimes.set_index('depTimes')
        # sort by arrival time
        edgeTimes = edgeTimes.sort_values('arrTimes')

        #Get trip times
        tripTimes = []
        for i in list(edgeTimes.index):
            tripTimes.append((3600 * i.hour) + (60 * i.minute) + i.second)
        tripTimes = np.array(tripTimes)

        # Create array of features
        tripFeatures = []
        for i, r in edgeTimes.iterrows():
            tripFeatures.append(
                [(3600 * r['arrTimes'].hour) + (60 * r['arrTimes'].minute) + r['arrTimes'].second, r['routeIDs'],
                 r['travelTime']])
        tripFeatures = np.array(tripFeatures)

        # G.add_edge(key[0],key[1],edgeFeature = edgeTimes)
        G.add_edge(key[0], key[1], edgeTripTimes = tripTimes, tripFeature=tripFeatures)
        success_count += 1
    else:
        fail_count += 1
        pass

#%%
#------------------ PART 4 - GET WALKING OA TO OA NODES ---------------------------------

# Get max and min lats for all data

network_type = 'walk'
trip_times = [5, 10, 15, 20, 25] #in minutes
travel_speed = 4.5 #walking speed in km/hour

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

GWalk = ox.graph.graph_from_bbox(north,south,east,west,network_type = 'all', retain_all=True)

# add an edge attribute for time in minutes required to traverse each edge
meters_per_minute = travel_speed * 1000 / 60 #km per hour to m per minute
for u, v, data in GWalk.edges(data=True):
    data['time'] = data['length'] / meters_per_minute

GWalk = ox.project_graph(GWalk, to_crs = 'WGS84')


meters_per_minute = travel_speed * 1000 / 60
oaIDIndex = list(oa_info['oa_id'])

nearestNodes = []

#for each OA get nearest node in network
count = 0
for oa in oaIDIndex:
    nextOA = (oa_info[oa_info['oa_id'] == oa]['oa_lat'].values[0],oa_info[oa_info['oa_id'] == oa]['oa_lon'].values[0])
    nearestNode = ox.get_nearest_node(GWalk, nextOA)
    nearestNodes.append(nearestNode)

oaToOaDist = np.zeros(shape=(len(oaIDIndex), len(oaIDIndex)))
shortestWalkingPaths = np.zeros(shape=(len(oaIDIndex), len(oaIDIndex)))

walkingMaxEuclidTime = 30
failRoutes = []
count_x = 0
for oa_x in nearestNodes:
    count_y = 0
    for oa_y in nearestNodes:
        count += 1
        if count % 100 == 0:
            percent_complete = (count / (len(nearestNodes) * len(nearestNodes))) * 100
            print(percent_complete)

        dist = haversine(oa_info.iloc[count_x]['oa_lon'], oa_info.iloc[count_x]['oa_lat'], oa_info.iloc[count_y]['oa_lon'], oa_info.iloc[count_y]['oa_lat'])
        oaToOaDist[count_x,count_y] = dist

        walkingTime = (dist * 1000) / meters_per_minute

        if walkingTime <= walkingMaxEuclidTime:
            try:
                dist, path = nx.single_source_dijkstra(GWalk, source=oa_x, target=oa_y, weight='time')
                shortestWalkingPaths[count_x, count_y] = dist
            except:
                failRoutes.append([oa_x,oa_y])

        count_y += 1
    count_x += 1

#manual fix to problem
#copy contents of oa 'E00048746' to oa 'E00048459'

shortestWalkingPaths[:,oaIDIndex.index('E00048746')] = shortestWalkingPaths[:,oaIDIndex.index('E00048459')]
shortestWalkingPaths[oaIDIndex.index('E00048746'),:] = shortestWalkingPaths[oaIDIndex.index('E00048459'),:]



#%%


#%%
np.savetxt(str(p) + '/Data/bus_network/OAtoOADist.csv', oaToOaDist, delimiter= ",")
np.savetxt(str(p) + '/Data/bus_network/OAtoOAWalkingTimes.csv', shortestWalkingPaths, delimiter= ",")




