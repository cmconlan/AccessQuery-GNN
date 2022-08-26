import numpy as np
import pandas as pd
import statsmodels.api as sm
import torch
from torch_geometric.nn import ChebConv
from torch.nn import Linear
from Scripts.expSetUp import appendPredictedCostToFeatures
import torch.nn.functional as F
import time

#%% Models

#Model 1- MLP
#source - https://medium.com/biaslyai/pytorch-introduction-to-neural-network-feedforward-neural-network-model-e7231cff47cb

class Feedforward(torch.nn.Module):
        def __init__(self, input_size, hidden_size):
            super(Feedforward, self).__init__()
            
            self.input_size = input_size
            self.hidden_size  = hidden_size
            
            self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
            self.fc2 = torch.nn.Linear(self.hidden_size, 1)
      
        def forward(self, x):
            hiddenLayer = self.fc1(x)
            hiddenActi = hiddenLayer.relu()
            
            hiddenActi = F.dropout(hiddenActi, p=0.1, training=self.training)
            
            outputLayer = self.fc2(hiddenActi)
            outputAct = outputLayer.relu()
            return outputAct

class GCN(torch.nn.Module):
    def __init__(self, numFeatures, hidden_channel1, hidden_channel2, outputDim, k, dp):
        super(GCN, self).__init__()
        torch.manual_seed(42)

        # Initialize the layers
        #GN Conv
        #self.conv1 = GCNConv(numFeatures, hidden_channel1, improved = True)
        #self.conv2 = GCNConv(hidden_channel1, hidden_channel2, improved = True)
        #Cheb Conv
        self.conv1 = ChebConv(numFeatures, hidden_channel1, K = k)
        self.conv2 = ChebConv(hidden_channel1, hidden_channel2, K = k)
        #SGConv
        #self.conv1 = SGConv(numFeatures, hidden_channel1)
        #self.conv2 = SGConv(hidden_channel1, hidden_channel2)
        #Cluster Conv
        #self.conv1 = ClusterGCNConv(numFeatures, hidden_channel1)
        #self.conv2 = ClusterGCNConv(hidden_channel1, hidden_channel2)
        self.out = Linear(hidden_channel2, outputDim)

    #def forward(self, x, edge_index):
    def forward(self, x, edge_index, edgeWeights, dp):
        # First Message Passing Layer (Transformation)
        x = self.conv1(x, edge_index, edgeWeights)
        #x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p= dp, training=self.training)

        # Second Message Passing Layer
        x = self.conv2(x, edge_index, edgeWeights)
        #x = self.conv2(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p= dp, training=self.training)
        
        # Output layer 
        x = self.out(x)
        x = x.relu()
        return x

#%%
#For each method
#Input variable depending on method
#Output an array of predicted values for the test set

#%% Method 1 - sampling

#Return the sampled values in base training data dataset

#Input - base trainnig, testMask

def simpleSampling(inputData, mask):
    timeStart = time.time()
    timeEnd = time.time()
    return inputData[mask]['sampleAccessCost'].values, timeEnd-timeStart

#predSample = simpleSampling(baseTraining, testMask)

#%% Method 2 - regression OLS

#Run OLS regression using features (not including predicted cost)

def OLSRegression(x,y,trainMask,testMask):
    
    timeStart = time.time()
    model = sm.OLS(y[trainMask], x[trainMask]).fit()
    
    predictedAccessCost = []
    for i in x[testMask]:
        predictedAccessCost.append(model.predict(i)[0])
    timeEnd = time.time()
    return np.array(predictedAccessCost), timeEnd-timeStart

#predOLS = OLSRegression(x,y,trainMask,testMask)

#%% Method 3 - regression MLP

def MLPRegression(x,y,trainMask,testMask,numHiddenLayers,epochs, device):
    timeStart = time.time()
    xTrain = torch.tensor(x[trainMask]).to(device).float()
    xTest = torch.tensor(x[testMask]).to(device).float()
    yTrain = torch.tensor(y[trainMask]).to(device).float()
    
    model = Feedforward(x.shape[1], numHiddenLayers)
    model.to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)
    
    losses = []
    
    model.train()
    for epoch in range(epochs):    
        optimizer.zero_grad()    # Forward pass
        y_pred = model(xTrain)    # Compute Loss
        loss = criterion(y_pred.squeeze(), yTrain)
        #print('Epoch {}: train loss: {}'.format(epoch, loss.item()))
        losses.append(loss)
        loss.backward()
        optimizer.step()

        # testTrain = float(losses[-1].cpu().detach().numpy()) - float(losses[0].cpu().detach().numpy())
        # if round(testTrain,3) != 0:
        #     proceed = True
    timeEnd = time.time()    
    return np.squeeze(model(xTest).cpu().detach().numpy()), timeEnd-timeStart, losses

#%% Method 4- Simple GNN

def GNNSimple(x,y,device,edgeIndexNp,edgeWeightsNp,hidden1,hidden2,epochs,trainMask,testMask,k, dp):
    timeStart = time.time()
    #Attach data to tensor
    _x = torch.tensor(x).to(device).float()
    _y = torch.tensor(y).to(device).float()
    edgeIndex = torch.tensor(edgeIndexNp).to(device).long()
    edgeWeights = torch.tensor(edgeWeightsNp).to(device).float()
    edgeWeights2 = torch.tensor(np.expand_dims(edgeWeightsNp,1)).to(device).float()
    
    #Instantitate moedl
    model = GCN(numFeatures = _x.shape[1], hidden_channel1=hidden1, hidden_channel2=hidden2, outputDim = 1, k=k, p=dp)
    model = model.to(device)
    
    #Model settings
    decay = 5e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=decay)
    criterion = torch.nn.MSELoss()

    model.train()
    
    losses = []
    
    for epoch in range(0, epochs):
        optimizer.zero_grad()
        out = model(_x, edgeIndex, edgeWeights2, dp)
        loss = criterion(out[trainMask], _y[trainMask].unsqueeze(1))
        loss.backward()
        optimizer.step()
        #print('Epoch {}: train loss: {}'.format(epoch, loss.item()))
        losses.append(loss.item())
    timeEnd = time.time()
    return np.squeeze(model(_x, edgeIndex, edgeWeights, dp)[testMask].cpu().detach().numpy()), timeEnd-timeStart, losses

#%%
#device = "cpu"
#hidden1 = 64
#hidden2 = 64
#epochs = 100
#predGNNSimple = GNNSimple(x,ySample,device,edgeIndexNp,edgeWeightsNp,hidden1,hidden2,epochs,trainMask,testMask)

#%% Method 5 - GNN with Seeds

#Train GNN with seeds

#_x = appendPredictedCostToFeatures(baseData,seedMask,features,x,target='sampleAccessCost')
#predGNNSeeds = GNNSimple(_x,ySample,device,edgeIndexNp,edgeWeightsNp,hidden1,hidden2,epochs,seedTrainMask,testMask)