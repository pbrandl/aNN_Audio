import torch
from torch import tensor
import torchaudio
import valveNN
from scipy import signal

model = valveNN.ValveNN(1, 400)

model.load_state_dict(torch.load("Models/first_try200929-1905"))

x_file = 'Audio/beat_raw.wav'
y_file = 'Audio/beat_pro.wav'

x_test, sr_x = torchaudio.load(x_file, normalization=True)
y_test, sr_y = torchaudio.load(y_file, normalization=True)

print("input file shape", x_test.shape)


def predict_sequence(x, input_size, window_size, offset, window=None):
    assert input_size == window_size + 2 * offset, "Window and offset has to match input of the model"

    d = x.shape[1] // window_size
    r = x.shape[1] % window_size

    pad_ending = torch.zeros(2, window_size - r + offset)
    pad_begining = torch.zeros(2, offset)

    x_padded = torch.cat((pad_begining, x, pad_ending), 1)

    print(d + 1, "==", (x_padded.shape[1] - 2 * offset) / window_size, '?')
    print(x_padded.shape)
    pred_seq = torch.tensor([])

    windows = range(offset, x.shape[1] + window_size, window_size)
    print(list(windows))
    for w_0, w_Z in zip(windows[:-1], windows[1:]):
        o_0 = w_0 - offset
        o_Z = w_Z + offset
        # print("o"+str(offset), "w"+str(w_0), str(w_Z), "o"+str(offset))

        pred = model(x_padded[0, o_0:o_Z].reshape(12000, 1, 1))

        pred_seq = torch.cat((pred_seq, pred[offset:offset + window_size, :, :]), dim=0)
        print(pred_seq.shape)

    return pred_seq.squeeze()[:-(window_size - r)]


output = predict_sequence(x_test, 12000, 10000, 1000, window=None)

print("Now writing to wav", output.shape)
torchaudio.save("Output/test_pred.wav", output, sr_x)

# infile_x = 'Data/trimmed_x.wav'
# infile_y = 'Data/trimmed_y.wav'
# data_x, sr_x = torchaudio.load(infile_x, normalization=True)
# data_y, sr_y = torchaudio.load(infile_y, normalization=True)

# assert sr_x == sr_y, "Expected audio data to be eqaul in sample rate."
# assert data_x.shape == data_y.shape, "Expected audio data to be eqaul in shape."

# start_data = 14641254
# end_data = data_x.shape[1]

# x_train = data_x[0, start_data:int(0.9*end_data)]
# y_train = data_y[0, start_data:int(0.9*end_data)]

# prediction = model(x_train[149*12000:149*12000+12000].reshape(12000, 1, 1))
# print(prediction.shape)
# print(sum(abs(prediction - y_train[149*12000:149*12000+12000].reshape(12000, 1, 1))))
