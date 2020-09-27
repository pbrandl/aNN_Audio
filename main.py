import numpy as np
import torchaudio
from torch import tensor, Tensor
from torch import nn
from torch.nn.modules.module import T_co


class ValveNN(nn.Module):
    def __init__(self):
        super().__init__()
        hidden_size = 1
        init_hidden_state = tensor(np.zeros(hidden_size))
        self.rnn_layer = nn.Sequential(nn.LSTM(input_size=1, hidden_size=1, num_layers=1))
        self.lin_layer = nn.Sequential(nn.Linear(in_features=hidden_size, out_features=1, bias=True))

        self.tanh = nn.Tanh()

    def forward(self, x) -> T_co:
        y = self.rnn_layer.forward(x)
        y = self.lin_layer.forward(y)
        y = self.tanh(y)

        return y


infile_x = 'Data/trimmed_x.wav'
infile_y = 'Data/trimmed_y.wav'
data_x, sr_x = torchaudio.load(infile_x)
data_y, sr_y = torchaudio.load(infile_y)

assert sr_x == sr_y, "Expected audio data to be eqaul in sample rate."
assert data_x.shape == data_y.shape, "Expected audio data to be eqaul in shape."

