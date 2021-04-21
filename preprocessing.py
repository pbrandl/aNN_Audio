import os
import numpy as np
import torchaudio
from torch import tensor


def arg_threshold(data: tensor, threshold: float = 0.8, offset: int = 50) -> (int, int):
    """
        This function returns the first and last indices of a 2D tensor that exceed a threshold.
    """

    assert 0 < data.shape[0] <= 2, "Expected mono/stereo audio data to be of form (channels, smaples)."

    start = 0
    end = len(data)
    for start, d in enumerate(data[0]):
        if abs(d) > threshold:
            print("Starts at: %i with volume %d." % (start, float(d)))
            break

    for i, d in enumerate(np.flipud(data[0])):
        if abs(d) > threshold:
            end = len(data[0]) - i
            print("Ends at: %i with volume %d." % (end, float(d)))
            break

    return start + offset, end - offset


def trim_audio_data(x_data, y_data, clip_offset=50):
    """
        This function trims two audio files that were previously embedded with two clip-sounds.
    """
    print("Original shape of X", x_data.shape)
    print("Original shape of Y", y_data.shape)

    start_x, end_x = arg_threshold(x_data, offset=clip_offset)
    start_y, end_y = arg_threshold(y_data, offset=clip_offset)
    x_trim = x_data[:, start_x:end_x]
    y_trim = y_data[:, start_y:end_y]

    print("Trimmed shape of X", x_trim.shape)
    print("Trimmed shape of Y", y_trim.shape)

    return x_trim, y_trim


# Load input x-file and target y-file
x_file = os.path.join("Data", "210317_Dataset_mono_x.wav")
y_file = os.path.join("Data", "210317_Dataset_mono_y.wav")
x_data, sr_x = torchaudio.load(x_file, normalization=True)
y_data, sr_y = torchaudio.load(y_file, normalization=True)
print(x_data.shape, y_data.shape)
assert sr_x == sr_y, "Expected sample rates of wav-files to be equal."
assert x_data.shape[1] <= y_data.shape[1], "x must be less-equal than y in order to trim."

x_trim, y_trim = trim_audio_data(x_data, y_data, clip_offset=51)

# !!Note: If difference is only 1 sample: The issue may be that the y dataset is too low in volume.!!
assert x_trim.shape == y_trim.shape, "Expected audio data to be eqaul in shape after trimming."

# Save preprocessed x- and y-files
x_file = os.path.join("Data", "Preprocessed", "210421_Dataset_mono_trim_x.wav")
y_file = os.path.join("Data", "Preprocessed", "210421_Dataset_mono_trim_y.wav")
torchaudio.save(x_file, x_trim, sr_x)
torchaudio.save(y_file, y_trim, sr_y)


