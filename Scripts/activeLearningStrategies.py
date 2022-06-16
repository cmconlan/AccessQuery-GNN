import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances_argmin_min

#%%

def basicClustering(b,inputData,oaIndex):

    clusterFeatures = np.array(inputData[['oa_lat','oa_lon']])
    
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(clusterFeatures)
    
    kmeans = KMeans(init="random",n_clusters=b,n_init=10,max_iter=300,random_state=42)
    kmeans.fit(scaled_features)
    
    closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, scaled_features)
    
    # oasSelected = []
    # for i in closest:
    #     oasSelected.append(oaIndex[i])
        
    return list(closest)

#%%

def distCluster(b,inputData):
    
    clusterFeatures = np.array(inputData[['oa_lat','oa_lon']])
    
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(clusterFeatures)
    
    kmeans = KMeans(init="random",n_clusters=b,n_init=10,max_iter=300,random_state=42)
    kmeans.fit(scaled_features)
    
    labels = kmeans.labels_
    
    oasSelected = []
    
    for i in range(b):
        itemindex = np.where(labels == i)
        oasSelected.append(inputData.iloc[itemindex]['avgEuclidDist'].idxmax())

    return oasSelected