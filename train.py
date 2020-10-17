import time

import torch
import torchaudio
from torch import nn

import valveNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

infile_x = 'Preprocessed/trimmed_x.wav'
infile_y = 'Preprocessed/trimmed_y.wav'
data_x, sr_x = torchaudio.load(infile_x, normalization=True)
data_y, sr_y = torchaudio.load(infile_y, normalization=True)

assert sr_x == sr_y, "Expected audio data to be eqaul in sample rate."
assert data_x.shape == data_y.shape, "Expected audio data to be eqaul in shape."

#assert y_train.shape == x_train.shape, "Expected train data to be in equal shape."

learning_rate = 1e-3
model = valveNN.ValveNN(input_size=1, hidden_size=400)
loss_fn = nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#model.load_state_dict(torch.load("Models/first_try200927-1921"))

start_time = time.time()

window_size = 12000
previous_size = 0

loss_history = []
epochs = 150
for epoch in range(epochs):
    optimizer.zero_grad()

    x = x_train[previous_size:previous_size+window_size].reshape(window_size, 1, 1)
    prediction = model(x)
    loss = loss_fn(y_train[previous_size:previous_size+window_size].reshape(window_size, 1, 1), prediction)

    previous_size += window_size
    loss_history.append(loss)
    if epoch % 5 == 0:
        print("Epoch", epoch, "Loss:", loss.item())

    loss.backward()
    optimizer.step()


print("Duration:", time.time() - start_time)

model_name = "first_try"
torch.save(model.state_dict(), "Models/first_try" + time.strftime("%y%m%d-%H%M"))


