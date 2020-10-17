import torch
from torch import nn


class ValveNN(nn.Module):
    def __init__(self, input_size=5, hidden_size=1):
        super().__init__()
        self.hs = torch.zeros(hidden_size)
        self.cs = torch.zeros(hidden_size)
        self.rnn_layer = nn.Sequential(nn.LSTM(input_size, hidden_size, num_layers=1))
        self.lin_layer = nn.Sequential(nn.Linear(in_features=hidden_size, out_features=1, bias=True))

        self.tanh = nn.Tanh()

    def forward(self, x):
        #print("Input X:", x.shape)
        y, (self.hs, self.cs) = self.rnn_layer.forward(x)
        #print("LSTM X:", y.shape)
        y = self.lin_layer(y)
        #print("Output X:", y.shape)

        return self.tanh(y)
