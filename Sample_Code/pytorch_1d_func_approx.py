# -*- coding: utf-8 -*-
__author__ = 'Gerald Schuller'
__copyright__ = 'G.S.'

"""
Simple program to use a fully connected linear neural network layer as a 1-dimensional faunction y=f(x) approximator.
This could be for instance an audio signal, x would be the time, y would be the audio signal value.
Gerald Schuller, Dec. 2019.
"""

import torch 
import torch.nn as nn
import matplotlib.pyplot as plt

device='cpu'
#device='cuda'

N=40 #number of samples in our signal
Nodes=10 #Number of Nodes in the hidden layer. This results in a smooth interpolation
#Nodes=80  #this can result in over-fitting

#A 2-layer fully connected network, 1 input (time value), hidden layer has "Nodes" nodes, 1 output (function value).

class LinNet(nn.Module):
    #define and initialize the layers:
    def __init__(self):
      super(LinNet, self).__init__()
      # Define the model. 
      #https://pytorch.org/docs/stable/nn.html?highlight=linear#torch.nn.Linear  
      # Generate a fully connected linear neural network model, 2 layers, bias
      # returns: Trainable object
      self.layer1=nn.Sequential(nn.Linear(in_features=1, out_features=Nodes, bias=True))
      self.layer2=nn.Sequential(nn.Linear(in_features=Nodes, out_features=1, bias=True))
      
      #self.act = nn.LeakyReLU() #non-linear activation function
      #self.act = nn.ReLU() #non-linear activation function
      #self.act = nn.Hardtanh() #non-linear activation function
      self.act = nn.Sigmoid() #non-linear activation function
      
    #Putting the network together:
    def forward(self, x):
      out = self.layer1(x)
      #print("out.shape=", out.shape)
      out = self.act(out)  #comment out if not desired
      #print("out.shape=", out.shape)
      out = self.layer2(out)
      #print("out.shape=", out.shape)
      return out
      
if __name__ == '__main__':
   print("Number of input samples:", N, "number of nodes:", Nodes)
   #input tensor, type torch tensor:
   #Indices: batch, additional dimensions, features or signal dimension. Here: 1 batch, 3 samples, signal dimension 2: 
   #Training set:
   #The x input here is the time:
   X=torch.arange(0,N,1.0) #generates N time steps for X
   X=X.view(N,1) #adding the last dimension for the signal (1 sample each), first dimension for the batch of size N
   print("X.shape", X.shape)
   #Target here is the (noisy) function value, a sine function + normal distributed random values:
   #Y=torch.sin(X)+torch.empty(X.shape).normal_(std=0.5)
   Y=torch.sin(X*3.14/N*2)+torch.randn(X.shape)*0.1
   #2 periods of the sinusoid in our training set, plus noise
   #Y=Y.view(N,1)
   print("Y.shape", Y.shape)
   #Validation set, to test generalization, with new noise:
   Xval=torch.arange(0.5,2*N,1.0) #generates 2N time steps for X for extrapolation beyond N,
   #shifted by 0.5 compared to training set, 
   #for interpolation between the original sample points.
   #print("Xval=", Xval)
   Xval=Xval.view(2*N,1)
   #Validation Target:
   Yval=torch.sin(Xval*3.14/N*2)+torch.randn(Xval.shape)*0.1
   #Yval=Yval.view(2*N,1)
   
   #create network object:
   model = LinNet().to(device)
   #Before training:
   Ypred=model(X) #the model produces prediction output
   print("Ypred.shape=", Ypred.shape)
   weights = model.state_dict()   #read obtained weights
   print("initial weights=", weights)  #see the random initialization of the weights 
   
   #print("model.parameters()=", model.parameters()) 
   
   print("Define loss function:")
   loss_fn = nn.MSELoss() #mean squared error loss
   
   print("Define optimizer:")
   #learning_rate = 1e-4
   optimizer = torch.optim.Adam(model.parameters())
   #optimizer = torch.optim.SGD(model.parameters(),lr=0.1)
   print("Optimizing:")
   for epoch in range(10000):
       Ypred=model(X) #the model produces prediction output
       loss=loss_fn(Ypred, Y) #prediction and target compared by loss
       if epoch%1000==0:
          print(epoch, loss.item()) #print current loss value
       optimizer.zero_grad() #optimizer sets previous gradients to zero
       loss.backward() #optimizer computes new gradients
       optimizer.step() #optimizer updates weights
       
   Ypred=model(X) # Make Predictions based on the obtained weights 
   #print("Ypred training set=", Ypred) 
   loss=loss_fn(Ypred, Y)
   print("Loss on trainig set:", loss.detach().numpy())
   plt.plot(X.detach().numpy()[:,0],Y.detach().numpy()[:,0])
   plt.plot(X.detach().numpy()[:,0],Ypred.detach().numpy()[:,0])
   plt.legend(('Training Target', 'Prediction Output'))
   plt.xlabel('X- Input')
   plt.ylabel('Y-Output')
   plt.title('Training Result')
   plt.show()
   Yvalpred=model(Xval) # Make Predictions based on the obtained weights 
   #print("Y validation set=", Yvalpred.detach().numpy()) 
   loss=loss_fn(Yvalpred[:N,:], Yval[:N,:])
   print("Loss on validation set:", loss.detach().numpy())
   plt.plot(Xval.detach().numpy()[:,0],Yval.detach().numpy()[:,0])
   plt.plot(Xval.detach().numpy()[:,0],Yvalpred.detach().numpy()[:,0])
   
   plt.xlabel('X- Input')
   plt.ylabel('Y-Output')
   plt.title('Generalization on Validation Set with Interpolation and Extrapolation')
   plt.legend(('Validation Target', 'Prediction Output'))
   plt.show()
   
   weights = model.state_dict()   #read obtained weights
   print("weights=", weights)
   #Weights of layer 2:
   layer2weights=model.state_dict()['layer2.0.weight'].clone() #clone(), otherwise it is just a pointer!
   print("model.state_dict()['layer2.0.weight']=", layer2weights)
   
   #The resulting function from the network is the sum of the functions of the N nodes.
   #To see the functions of the individual nodes, we can just keep their weight unchanged, 
   #set the others to zero, and plot the resulting function.
   #For that we make mask with zeros for the weights of layer 2, except for one node:
   
   for node in range(0,Nodes):
      weightmask=torch.zeros(layer2weights.shape) #mask with all zeros for output layer, except one.
      weightmask[0,node]=1.0 #node "node" unchanged
      #print("weightmask",weightmask)
      #print("layer2weights=",layer2weights)
      #print("layer2weights*weightmask=", layer2weights*weightmask)
      model.state_dict()['layer2.0.weight'].data.copy_(layer2weights*weightmask) #write pytorch structure back to model
      #print("Xval=", Xval)
      Ypred1node=model(Xval) # Make Predictions based on the 1-node weights 
      #print("Ypred1node=", Ypred1node)
      #Plot modified 1-node model:
      plt.plot(Xval.detach().numpy()[:,0],Ypred1node.detach().numpy()[:,0])
      
   #plt.legend(('Validation Target', 'Prediction Output', 'Node 0', 'Node 5'))
   plt.xlabel('X- Input')
   plt.ylabel('Y-Output')
   plt.title('Basis Functions of the Network')
   plt.legend(('Node 0', 'Node 1', 'Node2'))
   plt.show()
   #We see the activation function fit with bias and weight 
   #to different parts of the target function
