import numpy as np
import torchaudio


def split_in_batches():

def trim(data: np.array, start_threshold: float = 0.9, end_threshold: float = 0.9, offset: int = 50) -> (int, int):
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


def trim_audio_files(infile_x, inflie_y):
    infile_x = 'Data/train_x_clip.wav'
    infile_y = 'Data/train_y_clip.wav'
    data_x, sr_x = torchaudio.load(infile_x)
    data_y, sr_y = torchaudio.load(infile_y)

    assert sr_x == sr_y, "Expected audio data to be eqaul in sample rate."

    sample_rate = sr_x
    clip_offset = 50

    print("Original shape of X", data_x.shape)
    print("Original shape of Y", data_y.shape)

    start_x, end_x = trim(data_x, offset=clip_offset)
    start_y, end_y = trim(data_y, offset=clip_offset)
    x = data_x[:, start_x:end_x]
    y = data_y[:, start_y:end_y]

    print("Trimmed shape of X", x.shape)
    print("Trimmed shape of Y", y.shape)

    assert x.shape == y.shape, "Expected audio data to be eqaul in shape after trimming."

    torchaudio.save("Data/trimmed_x.wav", x, sr_x)
    torchaudio.save("Data/trimmed_y.wav", y, sr_y)
