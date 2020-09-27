# -*- coding: utf-8 -*-
__author__ = 'Gerald Schuller'
__copyright__ = 'G.S.'

"""
convolutional variational autoencoder for audio signals.
Denoising autoencoder is robust against noise at its input,
a variational autoencoder is robust against noise in its encoded domain.
Gerald Schuller, April 2020.
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
      #Analysis Filterbank with downsampling of N=8*1024, filter length of 2N, but only 32 outputs:
      #for the mean values:
      self.conv1mean=nn.Conv1d(in_channels=1, out_channels=32, kernel_size=8*2048, stride=8*1024, padding=8*1024-1, bias=True) #Padding for 'same' filters (kernel_size/2-1)
      #for the standard devieation values:
      self.conv1std=nn.Conv1d(in_channels=1, out_channels=32, kernel_size=8*2048, stride=8*1024, padding=8*1024-1, bias=True) #Padding for 'same' filters (kernel_size/2-1)

      #Synthesis filter bank:
      self.synconv1=nn.ConvTranspose1d(in_channels=32, out_channels=1, kernel_size=8*2048, stride=8*1024, padding=8*1024-1, bias=True)

   def encodermean(self, x):
      #Analysis:
      x = self.conv1mean(x)
      y = torch.tanh(x)
      return y
      
   def encoderstd(self, x):
      #Analysis:
      x = self.conv1std(x)
      y = torch.abs(torch.tanh(x))
      return y
      
   def decoder(self, y):
      #Synthesis:
      xrek= self.synconv1(y)
      return xrek
      
   def forward(self, x):
      Yencmean=model.encodermean(x)
      Yencstd=model.encoderstd(x)
      #Yvariational= torch.normal(Yencmean, Yencstd)
      Yvariational= Yencmean + Yencstd*torch.randn_like(Yencstd)
      #for the randn_like see also: https://github.com/pytorch/examples/blob/master/vae/main.py
      Ypred=model.decoder(Yvariational)
      return Ypred, Yencmean, Yencstd
    
def variational_loss(mu, std):
   #returns the varialtional loss from arguments mean and standard deviation std
   #see also: see Appendix B from VAE paper:
   # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
   #https://arxiv.org/abs/1312.6114
   vl=-0.5*torch.mean(1+ 2*torch.log(std)-mu.pow(2) -(std.pow(2)))
   return vl
   
def variational_loss2(mu, std):
   #returns the varialtional loss from arguments mean and standard deviation std
   #alternative: mean squared distance from ideal mu=0 and std=1:
   vl=torch.mean(mu.pow(2)+(1-std).pow(2))
   return vl

if __name__ == '__main__':
    #Testing:
    #compare the variational losses for different standard deviations std and mu=0:
    std=np.arange(0,20)
    vloss=torch.zeros(std.shape)
    vloss2=torch.zeros(std.shape)
    for std_ in std:
       vloss[std_]=variational_loss(torch.tensor([0.0]),torch.tensor(0.1*std_))
       vloss2[std_]=variational_loss2(torch.tensor([0.0]),torch.tensor([0.1*std_]))
    plt.plot(0.1*std,np.array(vloss))
    plt.plot(0.1*std,np.array(vloss2))
    plt.ylabel('Loss Value')
    plt.xlabel('Standard Deviation')
    plt.title('Variational Losses for mu=0')
    plt.legend(('Standard VAE Loss', 'Mean Squared Distance from ideal Loss'))
    plt.show()
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
    #make training set with batch of 2 speech signals:
    batch=2;
    fs, x0= wav.read('test2.wav') #get size of the speech files, all need to be identical
    xlen=max(x0.shape)
    x=np.zeros((batch,xlen))
    for b in range(batch):
       if b==0: 
          fs, x0= wav.read('test2.wav')
       if b==1:
          fs, x0= wav.read('test3.wav')
       x0= x0/max(x0)
       x[b,:]=x0
    #x=x*1.0/2**15 #normalize
    print("x.shape=", x.shape)
    X=signal2pytorch(x).to(device) #Convert to pytorch format, batch is first dimension    
    print("X.shape=", X.shape)
    print("Generate Model:")
    model = Convautoenc().to(device)
    print('Total number of parameters: %i' % (sum(p.numel() for p in model.parameters() if p.requires_grad)))
    print("Def. loss function:")
    loss_fn = nn.MSELoss()  #MSE
    #loss_fn = nn.L1Loss()
    
    Ypred, Yencmean, Yencstd = model(X)
    
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
       checkpoint = torch.load("audio_variational_autoenc.torch",map_location='cpu')
       model.load_state_dict(checkpoint['model_state_dict'])
       #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
       optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    except IOError:
       print("fresh start")
    """
    #optimrandomdir_pytorch.optimizer(model, loss_fn, X, Ypred, iterations=300, startingscale=1.0, endscale=0.0)
    #Ypred=model(X)
    #Ypred=Ypred.detach()
    #print("Ypred=", Ypred)
    
    #randdir=True # True for optimization of random direction, False for pytorch optimization
    randdir=False
    
    if randdir==True:
    #optimization of weights using method of random directions:
       optimrandomdir_pytorch.optimizer(model, loss_fn, X, Y, iterations=100000, startingscale=0.25, endscale=0.0)
       #--End optimization of random directions------------------------
    else:
       for epoch in range(2000):
          #Ypred, Yencmean, Yencstd = model(X)
          #mean values from the encoder network:
          Yencmean=model.encodermean(X)
          
          #standard deviation values from the network:
          Yencstd=model.encoderstd(X)
          #unit standard deviation:
          #Yencstd=torch.ones(Yencmean.shape)
          
          Yvariational= Yencmean + Yencstd*torch.randn_like(Yencstd)
          Ypred=model.decoder(Yvariational)
          
          mse=loss_fn(Ypred, Y)
          vl=variational_loss(Yencmean, Yencstd)
          #vl=variational_loss2(Yencmean, Yencstd)
          loss= mse + 0.01*vl
          #loss= mse 
          if epoch%10==0:
             print(epoch, "mse=", mse.item(), "variational loss=", vl.item())
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
    #"""
    torch.save({#'epoch': epoch,
       'model_state_dict': model.state_dict(),
       'optimizer_state_dict': optimizer.state_dict()}, "audio_variational_autoenc.torch")
    #"""
    print("MSE=", loss_fn(Ypred, Y).item(), "Variational Loss:", variational_loss(Yencmean, Yencstd).item())
    ww = model.state_dict()   #read obtained weights
    print("ww=", ww)
    #Plot obtained weights:
    plt.plot(np.transpose(np.array(ww['conv1mean.weight'][0:1,0,:])))
    plt.plot(np.transpose(np.array(ww['conv1std.weight'][0:1,0,:])))
    plt.plot(np.transpose(np.array(ww['synconv1.weight'][0:1,0,:])))
    plt.legend(('Encoder Analysis filter 0, mean','Encoder Analysis filter 0, std', 'Decoder Filter 0'))
    plt.xlabel('Sample')
    plt.ylabel('Value')
    plt.title('The Encoder and Decoder Filter Coefficients')
    plt.show()
    
    #Test on training set:
    
    #predictions=model(X).cpu() # Make Predictions based on the obtained weights, on training set
    #noisy case:
    #predictions, Yencmean, Yencstd = model(X)
    Yencmean=model.encodermean(X)
    
    #no noise case
    predclean=model.decoder(Yencmean)
    predclean=predclean.detach() #no noise case
    predclean=np.array(predclean)
    
    #Add gaussian noise with unit standard deviation to encoded signal:
    Yvariational= Yencmean + torch.randn_like(Yencmean)
    predictions=model.decoder(Yvariational)
    
    predictions=predictions.detach()
    predictions=np.array(predictions)
    Yencmean=np.array(Yencmean.detach())
    Yencstd=np.array(Yencstd.detach())
    print("Yencstd.shape=",Yencstd.shape)
    
    Y=np.array(Y) #target
    #print("Y=",Y)
    print("predictions.shape=", predictions.shape)
    #convert to numpy:
    #https://discuss.pytorch.org/t/how-to-transform-variable-into-numpy/104/2
    #Plot target signal and output of autoencoder:
    for b in range(batch):
       #print("np.reshape(Yencstd[b,:,:],(1,-1))", np.reshape(Yencstd[b,:,:],(1,-1)))
       plt.plot(np.reshape(Yencmean[b,:,:],(1,-1))[0,:])
       plt.plot(np.reshape(Yencstd[b,:,:],(1,-1))[0,:])
       plt.legend(('Encoded Mean', 'Encoded Standard Deviation'))
       plt.title('The Encoded Domain, Mean and Standard Deviation')
       plt.show()
       plt.plot(np.array(Y[b,0,:]))
       plt.plot(predictions[b,0,:])
       plt.legend(('Target','Predicted'))
       plt.title('The Target and Noisy Predicted Signal, batch '+str(b))
       plt.xlabel('Sample')
       plt.show()

       #No noise case:
       xrek=predclean[b,0,:]  #remove unnecessary dimension for playback
       xrek=np.clip(xrek, -1.0,1.0)
       wav.write('testrekvaeclean'+str(b)+'.wav', fs, np.int16(2**15*xrek))
       os.system('espeak -ven -s 120 '+'"The training set output for clean encoded signal for batch'+str(b)+'"')
       sound(2**15*xrek,fs)
       
       xrek=predictions[b,0,:]  #remove unnecessary dimension for playback
       xrek=np.clip(xrek, -1.0,1.0)
       wav.write('testrekvae'+str(b)+'.wav', fs, np.int16(2**15*xrek))
       os.system('espeak -ven -s 120 '+'"The training set output for noisy encoded signal for batch'+str(b)+'"')
       sound(2**15*xrek,fs)
    
    
    #Test on Verification set:
    fs, x= wav.read('test.wav')
    #fs, x= wav.read('test2.wav')
    #x=x*1.0/2**15 #normalize
    x=x/max(x)
    
    os.system('espeak -ven -s 120 '+'"The verification set input to the variational autoencoder"')
    sound(2**14*x,fs)
    X=signal2pytorch(x).to(device)
    Yencmean=model.encodermean(X)

    predclean=model.decoder(Yencmean)
    predclean=predclean.detach() #no noise case
    predclean=np.array(predclean)
    
    #No noise case:
    xrek=predclean[0,0,:]  #remove unnecessary dimension for playback
    xrek=np.clip(xrek, -1.0,1.0)
    wav.write('testvervaeclean'+str(b)+'.wav', fs, np.int16(2**15*xrek))
    os.system('espeak -ven -s 120 '+'"The verification set output for clean encoded signal"')
    sound(2**15*xrek,fs)
    
    #Add gaussian noise with unit standard deviation:
    Yvariational= Yencmean + torch.randn_like(Yencmean)
    predictions= model.decoder(Yvariational) # Make Predictions based on the obtained weights, on verification set
    predictions=predictions.detach()
    predictions=np.array(predictions)
    b=0
    plt.plot(np.array(X[b,0,:]))
    plt.plot(predictions[b,0,:])
    plt.legend(('Original','Predicted'))
    plt.title('Verification, the Original and Predicted Signal, batch ')
    plt.xlabel('Sample')
    plt.show()
    xrek=predictions[:,0,:]
    xrek=np.transpose(xrek)
    xrek=np.clip(xrek, -1.0,1.0)
    wav.write('testver.wav', fs, np.int16(2**15*xrek))
    os.system('espeak -ven -s 120 '+'"The verification set output for noisy encoded signal"')
    sound(2**14*xrek,fs)
    
    os.system('espeak -ven -s 120 '+'"Only noise as encoded signal"')
    Yvariational=  torch.randn_like(Yencmean)

    predictions= model.decoder(Yvariational) # Make Predictions based on the obtained weights, on verification set
    predictions=predictions.detach()
    predictions=np.array(predictions)
    xrek=predictions[:,0,:]
    xrek=np.transpose(xrek)
    xrek=np.clip(xrek, -1.0,1.0)
    wav.write('noisevarout.wav', fs, np.int16(2**15*xrek))
    os.system('espeak -ven -s 120 '+'"The decoded signal"')
    sound(2**14*xrek,fs)
