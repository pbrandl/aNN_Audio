# -*- coding: utf-8 -*-
__author__ = 'Gerald Schuller'
__copyright__ = 'G.S.'

"""
convolutional autoencoder for audio signals.
Gerald Schuller, February 2020.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import scipy.io.wavfile as wav
from sound import sound

#import optimrandomdir_pytorch 
  
if sys.version_info[0] < 3:
   # for Python 2
   import cPickle as pickle
else:
   # for Python 3
   import pickle
   
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device=", device)   



def signal2pytorch(x):
   #Function to convert a signal vector x, like a mono audio signal, into a 3-d Tensor that conv1d of Pytorch expects,
   #https://pytorch.org/docs/stable/nn.html
   #Argument x: a 1-d signal as numpy array
   #input x[batch,sample]
   #output: 3-d Tensor X for conv1d input.
   #for conv1d Input: (N,Cin,Lin), Cin: numer of input channels (e.g. for stereo), Lin: length of signal, N: number of Batches (signals) 
   X = np.expand_dims(x, axis=0)  #add channels dimension (here only 1 channel)
   if len(x.shape)==1: #mono:
      X = np.expand_dims(X, axis=0)  #add batch dimension (here only 1 batch)
   X=torch.from_numpy(X)
   X=X.type(torch.Tensor)
   X=X.permute(1,0,2)  #make batch dimension first
   return X

class Convautoenc(nn.Module):
   def __init__(self):
      super(Convautoenc, self).__init__()
      #Analysis Filterbank with downsampling of N=1024, filter length of 2N, but only N/2 outputs:
      self.conv1=nn.Conv1d(in_channels=1, out_channels=32, kernel_size=2048, stride=1024, padding=1023, bias=True) #Padding for 'same' filters (kernel_size/2-1)

      #Synthesis filter bank:
      self.synconv1=nn.ConvTranspose1d(in_channels=32, out_channels=1, kernel_size=2048, stride=1024, padding=1023, bias=True)

   def encoder(self, x):
      #Analysis:
      x = self.conv1(x)
      y = torch.tanh(x)
      return y
      
   def decoder(self, y):
      #Synthesis:
      xrek= self.synconv1(y)
      return xrek
      
   def forward(self, x):
      y=self.encoder(x)
      #y=torch.round(y/0.125)*0.125
      xrek=self.decoder(y)
      return xrek

    

if __name__ == '__main__':
    #Testing:
    
    """
    batch=2 #number of audio files in the batch
    fs, audio = wav.read('fantasy-orchestra.wav')
    audio0=audio[:,0] #make it mono left channel
    audio1=audio[:,1] #make it mono right channel
    #audio=audio*1.0/2**15 #normalize
    audio0=audio0*1.0/np.max(np.abs(audio0)) #normalize
    audio1=audio1*1.0/np.max(np.abs(audio1)) #normalize
    print("audio0.shape=", audio0.shape)
    #audiosh=audio[46750:58750] #shorten the signal for faster optimization,
    #audiosh=audio[10000:60000] 
    audiolen=100000
    x=np.zeros((batch, audiolen))
    x[0,:]=audio0[np.arange(audiolen)] 
    x[1,:]=audio1[np.arange(audiolen)]
    """
    #alternative: speech:
    batch=1
    fs, x= wav.read('test.wav')
    #fs, x= wav.read('test2.wav')
    #x=x*1.0/2**15 #normalize
    x= x/max(x)
    
    X=signal2pytorch(x).to(device) #Convert to pytorch format, batch is first dimension    
    
    print("Generate Model:")
    model = Convautoenc().to(device)
    print('Total number of parameters: %i' % (sum(p.numel() for p in model.parameters() if p.requires_grad)))
    print("Def. loss function:")
    loss_fn = nn.MSELoss()  #MSE
    #loss_fn = nn.L1Loss()
    
    Ypred=model(X)
   
    #Ypred=Ypred.detach()
    outputlen=len(Ypred[0,0,:]) #length of the signal at the output of the network.
    print("outputlen=", outputlen)
    
    Y=X[:,:,:outputlen]  #the target signal with same length as model output
    
    print("Input X.shape=", X.shape )
    print("Target Y.shape=", Y.shape)
    print("Target Y=", Y)
    #print("max(max(Y))=", max(max(max(Y))))
    #print("min(min(Y))=", min(min(min(Y))))
    print("Y.type()=", Y.type())
    
    
    learning_rate = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)#, betas=(0.9, 0.999))
    """
    try:
       checkpoint = torch.load("audio_autoenc.torch",map_location='cpu')
       model.load_state_dict(checkpoint['model_state_dict'])
       #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
       optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    except IOError:
       print("fresh start")
    """
    #optimrandomdir_pytorch.optimizer(model, loss_fn, X, Ypred, iterations=300, startingscale=1.0, endscale=0.0)
    Ypred=model(X)
    #Ypred=Ypred.detach()
    print("Ypred=", Ypred)
    
    #randdir=True # True for optimization of random direction, False for pytorch optimization
    randdir=False
    
    if randdir==True:
    #optimization of weights using method of random directions:
       optimrandomdir_pytorch.optimizer(model, loss_fn, X, Y, iterations=100000, startingscale=0.25, endscale=0.0)
       #--End optimization of random directions------------------------
    else:
       for epoch in range(2000):
          Ypred=model(X)
          #print("Ypred.shape=", Ypred.shape)
          #loss wants batch in the beginning! (Batch, Classes,...)
          #Ypredp=Ypred.permute(1,2,0)
          #Yp=Y.permute(1,0)
          #print("Ypredp.shape=", Ypredp.shape, "Yp.shape=", Yp.shape )
          loss=loss_fn(Ypred, Y)
          if epoch%10==0:
             print(epoch, loss.item())
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
    """
    torch.save({#'epoch': epoch,
       'model_state_dict': model.state_dict(),
       'optimizer_state_dict': optimizer.state_dict()}, "audio_autoenc.torch")
    """
    
    ww = model.state_dict()   #read obtained weights
    print("ww=", ww)
    #Plot obtained weights:
    plt.plot(np.transpose(np.array(ww['conv1.weight'][0:1,0,:])))
    plt.plot(np.transpose(np.array(ww['synconv1.weight'][0:1,0,:])))
    plt.legend(('Encoder Analysis filter 0', 'Decoder Filter 0'))
    plt.xlabel('Sample')
    plt.ylabel('Value')
    plt.title('The Encoder and Decoder Filter Coefficients')
    plt.show()
    
    #Test on training set:
    predictions=model(X).cpu() # Make Predictions based on the obtained weights, on training set
    predictions=predictions.detach()
    predictions=np.array(predictions)
    Y=np.array(Y) #target
    #print("Y=",Y)
    print("predictions.shape=", predictions.shape)
    #convert to numpy:
    #https://discuss.pytorch.org/t/how-to-transform-variable-into-numpy/104/2
    #Plot target signal and output of autoencoder:
    for b in range(batch):
       plt.plot(np.array(Y[b,0,:]))
       plt.plot(predictions[b,0,:])
       plt.legend(('Target','Predicted'))
       plt.title('The Target and Predicted Signal, batch '+str(b))
       plt.xlabel('Sample')
       plt.show()
    xrek=predictions[:,0,:]  #remove unnecessary dimension for playback
    xrek=np.transpose(xrek)
    xrek=np.clip(xrek, -1.0,1.0)
    wav.write('testrek.wav', fs, np.int16(2**15*xrek))
    os.system('espeak -ven -s 120 '+'"The training set output of the autoencoder"')
    sound(2**15*xrek,fs)
    
    #Test on Verification set:
    fs, x= wav.read('test2.wav')
    #fs, x= wav.read('test.wav')
    #x=x*1.0/2**15 #normalize
    x=x/max(x)
    X=signal2pytorch(x).to(device)
    predictions=model(X).cpu() # Make Predictions based on the obtained weights, on verification set
    predictions=predictions.detach()
    predictions=np.array(predictions)
    for b in range(batch):
       plt.plot(np.array(X[b,0,:]))
       plt.plot(predictions[b,0,:])
       plt.legend(('Original','Predicted'))
       plt.title('The Original and Predicted Signal, batch '+str(b))
       plt.xlabel('Sample')
       plt.show()
    xrek=predictions[:,0,:]
    xrek=np.transpose(xrek)
    xrek=np.clip(xrek, -1.0,1.0)
    wav.write('test2rek.wav', fs, np.int16(2**15*xrek))
    os.system('espeak -ven -s 120 '+'"The verification set output of the autoencoder"')
    sound(2**15*xrek,fs)
    
    #Test on shifted input:
    fs, x= wav.read('test.wav')
    #fs, x= wav.read('test2.wav')
    #x=x*1.0/2**15 #normalize
    x=x/max(x)
    x=np.append(np.zeros(100),x) #prepend zeros to test time or shift invariance
    X=signal2pytorch(x).to(device)
    predictions=model(X).cpu() # Make Predictions based on the obtained weights, on verification set
    predictions=predictions.detach()
    predictions=np.array(predictions)
    xrek=predictions[:,0,:]
    xrek=np.transpose(xrek)
    xrek=np.clip(xrek, -1.0,1.0)
    os.system('espeak -ven -s 120 '+'"The 100 samples shifted traings set output of the autoencoder"')
    sound(2**14*xrek,fs)
    
    #Test on 1024 samples shifted test set (shift identical to the stride size)
    fs, x= wav.read('test.wav')
    #fs, x= wav.read('test2.wav')
    #x=x*1.0/2**15 #normalize
    x=x/max(x)
    x=np.append(np.zeros(1024),x) #prepend zeros to test time or shift invariance
    X=signal2pytorch(x).to(device)
    predictions=model(X).cpu() # Make Predictions based on the obtained weights, on verification set
    predictions=predictions.detach()
    predictions=np.array(predictions)
    xrek=predictions[:,0,:]
    xrek=np.transpose(xrek)
    xrek=np.clip(xrek, -1.0,1.0)
    os.system('espeak -ven -s 120 '+'"The 1024 samples shifted traings set output of the autoencoder"')
    sound(2**14*xrek,fs)
    
