import os
import numpy as np
import torchaudio
from torch import tensor
import matplotlib.pyplot as plt

def trim(data: tensor, start_threshold: float = 0.9, end_threshold: float = 0.9, offset: int = 50) -> (int, int):
    assert data.shape[0] == 2, "Expected shape of audio data to be of form (2, n)."

    start = 0
    end = len(data)
    for start, d in enumerate(data[0]):
        if abs(d) > start_threshold:
            print("Starts at: %i with volume %d." % (start, float(d)))
            break

    for i, d in enumerate(np.flipud(data[0])):
        if abs(d) > end_threshold:
            end = len(data[0]) - i
            print("Ends at: %i with volume %d." % (end, float(d)))
            break

    return start + offset, end - offset


def trim_audio_data(x_data, y_data, clip_offset=50):
    print("Original shape of X", x_data.shape)
    print("Original shape of Y", y_data.shape)

    start_x, end_x = trim(x_data, offset=clip_offset)
    start_y, end_y = trim(y_data, offset=clip_offset)
    x_trim = x_data[:, start_x:end_x]
    y_trim = y_data[:, start_y:end_y]

    print("Trimmed shape of X", x_trim.shape)
    print("Trimmed shape of Y", y_trim.shape)

    return x_trim


x_file = os.path.join("Data", "x_rand_train_18000.wav")
y_file = os.path.join("Data", "y_rand_train_18000.wav")

x_data, sr_x = torchaudio.load(x_file, normalization=True)
y_data, sr_y = torchaudio.load(y_file, normalization=True)

assert sr_x == sr_y, "Expected sample rates of wav-files to be equal."

#x_trim, y_trim = trim_audio_data(x_data, y_data, 50)
x_trim = x_data[:, 50:-50]
y_trim = y_data[:, 339566:90339566]
assert x_trim.shape == y_trim.shape, "Expected audio data to be eqaul in shape after trimming."

torchaudio.save("Preprocessed/trimmed_x.wav", x_trim, sr_x)
torchaudio.save("Preprocessed/trimmed_y.wav", y_trim, sr_y)
