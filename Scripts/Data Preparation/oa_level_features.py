import geopandas as gpd
import pandas as pd
from pathlib import Path
import shapely as sp
from functools import partial
import pyproj
from shapely.geometry import Point
from shapely.ops import transform
from statistics import mean

proj_wgs84 = pyproj.Proj('+proj=longlat +datum=WGS84')

def geodesic_point_buffer(lat, lon, km):
    # Azimuthal equidistant projection
    aeqd_proj = '+proj=aeqd +lat_0={lat} +lon_0={lon} +x_0=0 +y_0=0'
    project = partial(
        pyproj.transform,
        pyproj.Proj(aeqd_proj.format(lat=lat, lon=lon)),
        proj_wgs84)
    buf = Point(0, 0).buffer(km * 1000)  # distance in metres
    return transform(project, buf).exterior.coords[:]


# def get_features_for_stop(stop_ids):

#     number_of_routes = 0
#     number_of_buses = 0
#     number_low_frequency_routes = 0
#     routes = []
#     route_frequencies = []
#     max_frequencies = []
#     count_inactive = 0
#     for id_to_test in stop_ids:
#         stop_time_for_id = stop_times[stop_times['stop_id'] == id_to_test]
#         stop_time_with_trips = stop_time_for_id.merge(trips[['route_id', 'trip_id', 'service_id']], how="inner",
#                                                       left_on='trip_id', right_on='trip_id')
#         stop_time_with_dates = stop_time_with_trips.merge(calendar, left_on='service_id', right_on='service_id',
#                                                           how='inner')
#         all_bus_service_tuesday = stop_time_with_dates[stop_time_with_dates['tuesday'] == 1][
#             ['arrival_time', 'departure_time', 'stop_id', 'route_id']]

#         if all_bus_service_tuesday.shape[0] == 0:
#             count_inactive += 1

#         else:

#             route_frequency = all_bus_service_tuesday['route_id'].value_counts()
#             low_frequency = route_frequency[route_frequency.le(5)]
#             max_frequency = route_frequency.max()
#             number_of_routes += route_frequency.shape[0]
#             number_of_buses += all_bus_service_tuesday.shape[0]
#             number_low_frequency_routes += low_frequency.shape[0]
#             routes = routes + list(route_frequency.index)
#             route_frequencies = route_frequencies + list(route_frequency.values)
#             max_frequencies.append(max_frequency)

#     if len(stop_ids) - count_inactive > 0:

#         distinct_number_of_routes = len(set(routes))
#         number_of_buses
#         avg_routes_per_stop = number_of_routes / (len(stop_ids) - count_inactive)
#         max_route_frequency = max(route_frequencies)
#         average_route_frequency = mean(route_frequencies)
#         average_max_frequency = mean(max_frequencies)
#         percent_low_frequency_routes = number_low_frequency_routes / number_of_routes

#         features = [distinct_number_of_routes, number_of_buses, avg_routes_per_stop, max_route_frequency,
#                     average_route_frequency, average_max_frequency, percent_low_frequency_routes]
#         return features

#     else:
#         return None

def get_features_for_stop(stop_ids):
    
    count_inactive = 0
    number_of_routes = 0
    number_of_buses = 0
    routes = []
    
    for id_to_test in stop_ids:
        stop_time_for_id = stop_times[stop_times['stop_id'] == id_to_test]
        stop_time_with_trips = stop_time_for_id.merge(trips[['route_id', 'trip_id', 'service_id']], how="inner",
                                                      left_on='trip_id', right_on='trip_id')
        stop_time_with_dates = stop_time_with_trips.merge(calendar, left_on='service_id', right_on='service_id',
                                                          how='inner')
        all_bus_service_tuesday = stop_time_with_dates[stop_time_with_dates['tuesday'] == 1][
            ['arrival_time', 'departure_time', 'stop_id', 'route_id']]

        if all_bus_service_tuesday.shape[0] == 0:
            count_inactive += 1

        else:            
            route_frequency = all_bus_service_tuesday['route_id'].value_counts()
            number_of_routes += route_frequency.shape[0]
            number_of_buses += all_bus_service_tuesday.shape[0]
            routes = routes + list(route_frequency.index)
    
    numActiveStop = (len(stop_ids) - count_inactive)
    
    #number active bus stops, number of routes, number of buses
    return numActiveStop, number_of_routes, number_of_buses 
    
#%%

p = Path(__file__).parents[2]

gtfs_data = str(p) + '/Data/tfwm_gtfs/'

calendar = pd.read_csv(gtfs_data+'calendar.txt', sep=",", header=0)
calendar_dates = pd.read_csv(gtfs_data+'calendar_dates.txt', sep=",", header=0)
routes = pd.read_csv(gtfs_data+'routes.txt', sep=",", header=0)
stop_times = pd.read_csv(gtfs_data+'stop_times.txt', sep=",", header=0, dtype={3:str,5:str})
stops = pd.read_csv(gtfs_data+'stops.txt', sep=",", header=0)
trips = pd.read_csv(gtfs_data+'trips.txt', sep=",", header=0)

wm_oas = gpd.read_file(str(p) + '/Data/west_midlands_OAs/west_midlands_OAs.shp')

#select coventry
wm_oas = wm_oas[wm_oas['LAD11CD'] == 'E08000026']
oa_info = pd.read_csv(str(p) + '/Data/oa_info.csv')

wm_oas = wm_oas.merge(oa_info[['oa_id','oa_lat','oa_lon']], left_on = 'OA11CD', right_on = 'oa_id', how = 'left')

oaIDIndex = list(oa_info['oa_id'])

#%%

stops_geo_points = []

for i,r in stops.iterrows():
    stops_geo_points.append(sp.geometry.Point(r['stop_lon'], r['stop_lat']))

stops['Geo Points'] = stops_geo_points

#%%

wm_oas['Count Bus Stops in OA'] = 0
wm_oas['Stop IDs in OA'] = None
wm_oas['OA Level Features'] = None

wm_oas['Count Bus Stops within 200m'] = 0
wm_oas['Stop IDs within 200m'] = None
wm_oas['200m Level Features'] = None

wm_oas['Count Bus Stops within 500m'] = 0
wm_oas['Stop IDs within 500m'] = None
wm_oas['500m Level Features'] = None

wm_oas['Count Bus Stops within 1km'] = 0
wm_oas['Stop IDs within 1km'] = None
wm_oas['1km Level Features'] = None

#%%
count_oa = 0

for i,r in wm_oas.iterrows():
    count_oa += 1
    print('Iteration : ' + str(count_oa))
    oa_shape = sp.geometry.asShape(r['geometry'])
    radius_200m = sp.geometry.asShape(sp.geometry.Polygon(geodesic_point_buffer(r['oa_lat'], r['oa_lon'], 0.2)))
    radius_500m = sp.geometry.asShape(sp.geometry.Polygon(geodesic_point_buffer(r['oa_lat'], r['oa_lon'], 0.5)))
    radius_1km = sp.geometry.asShape(sp.geometry.Polygon(geodesic_point_buffer(r['oa_lat'], r['oa_lon'], 1)))

    count_stops_in_oa = 0
    stop_ids_in_oa = []

    count_stops_200m = 0
    stop_ids_200m = []

    count_stops_500m = 0
    stop_ids_500m = []

    count_stops_1km = 0
    stop_ids_1km = []

    for i_stop,r_stops in stops.iterrows():
        if oa_shape.contains(r_stops['Geo Points']):
            count_stops_in_oa += 1
            stop_ids_in_oa.append(r_stops['stop_id'])

        if radius_200m.contains(r_stops['Geo Points']):
            count_stops_200m += 1
            stop_ids_200m.append(r_stops['stop_id'])

        if radius_500m.contains(r_stops['Geo Points']):
            count_stops_500m += 1
            stop_ids_500m.append(r_stops['stop_id'])

        if radius_1km.contains(r_stops['Geo Points']):
            count_stops_1km += 1
            stop_ids_1km.append(r_stops['stop_id'])

    wm_oas.loc[i,'Count Bus Stops in OA'] = count_stops_in_oa
    wm_oas.at[i,'Stop IDs in OA'] = stop_ids_in_oa
    if len(stop_ids_in_oa)> 0:
        wm_oas.at[i, 'OA Level Features'] = get_features_for_stop(stop_ids_in_oa)

    wm_oas.loc[i,'Count Bus Stops within 200m'] = count_stops_200m
    wm_oas.at[i,'Stop IDs within 200m'] = stop_ids_200m
    if len(stop_ids_200m) > 0:
        wm_oas.at[i, '200m Level Features'] = get_features_for_stop(stop_ids_200m)

    wm_oas.loc[i,'Count Bus Stops within 500m'] = count_stops_500m
    wm_oas.at[i,'Stop IDs within 500m'] = stop_ids_500m
    if len(stop_ids_500m) > 0:
        wm_oas.at[i, '500m Level Features'] = get_features_for_stop(stop_ids_500m)

    wm_oas.loc[i,'Count Bus Stops within 1km'] = count_stops_1km
    wm_oas.at[i,'Stop IDs within 1km'] = stop_ids_1km
    if len(stop_ids_1km) > 0:
        wm_oas.at[i, '1km Level Features'] = get_features_for_stop(stop_ids_1km)

#%%
wm_oas.to_csv(str(p) + '/Data/bus_network/OALevelFeatures.csv') 

