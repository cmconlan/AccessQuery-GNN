import numpy as np
import networkx as nx
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

#%%

def degreeCentrality(b,adjMx):
    G = nx.DiGraph(adjMx)
    nodesL = []

    for i in range(b):
        dc = nx.degree_centrality(G)
        dcSort = dict(sorted(dc.items(), key=lambda item: item[1], reverse=True))
        nextNode = next(iter(dcSort))
        nodesL.append(nextNode)
        del dcSort[nextNode]
        G.remove_node(nextNode)

    return nodesL

#%%

def eigenCentrality(b,adjMx):
    G = nx.DiGraph(adjMx)
    nodesL = []

    for i in range(b):
        dc = nx.eigenvector_centrality_numpy(G, weight = 'weight')
        dcSort = dict(sorted(dc.items(), key=lambda item: item[1], reverse=True))
        nextNode = next(iter(dcSort))
        nodesL.append(nextNode)
        del dcSort[nextNode]
        G.remove_node(nextNode)

    return nodesL

#%%

#cluster feature space
def featureCluster(b,inputData,mf):
    
    clusterFeatures = np.array(inputData[mf])
    
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(clusterFeatures)
    
    kmeans = KMeans(init="random",n_clusters=b,n_init=10,max_iter=300,random_state=42)
    kmeans.fit(scaled_features)
    
    scaled_features = scaled_features.copy(order='C')
    
    closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, scaled_features)
    
    return list(closest)

#%%
#cluster node embedding

def embedCluster(b,area,embedType = 'gauss'):
    
    if embedType == 'euclid':
        clusterFeatures = np.load('Data/adjMx/' + str(area) + '/euclidEmbeddings.csv')
    elif embedType == 'gauss':
        clusterFeatures = np.load('Data/adjMx/' + str(area) + '/adjMxEmbeddings.csv')
    
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(clusterFeatures)
    
    kmeans = KMeans(init="random",n_clusters=b,n_init=10,max_iter=300,random_state=42)
    kmeans.fit(scaled_features)
    
    scaled_features = scaled_features.copy(order='C')
    
    closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, scaled_features)
    
    return list(closest)






























