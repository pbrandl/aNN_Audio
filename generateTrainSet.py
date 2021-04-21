import os
import glob
import random
import torch
import torchaudio
from scipy import signal


def dataset_generator(files, n_slice, slice_size, x_fade=0, channels=1):
    """
        This function randomly cuts slices from a song collections. Slices are concatenated
        and returned as a long tensor of audio slices. Slices are faded to avoid click-noise.
        A short and loud clip is added to the start and end of the final tensor. This ensures
        that the input and target data can be synchronized after recording.
    """

    # Allocate Torch Tensor
    x = torch.zeros(channels, slice_size * n_slice + x_fade)

    # Initialise Tukey Window
    win_size = slice_size + x_fade // 2
    alpha = 0 if x_fade == 0 else x_fade / win_size
    tukey_win = torch.tensor(signal.tukey(win_size, alpha, sym=False))

    print('Tukey window satisfies COLA:', signal.check_COLA(tukey_win, win_size, x_fade // 2))

    # Add a Random Slice of a Random Song to Train Data x
    for i, wav_file in enumerate(random.choices(files, k=n_slice), 0):
        if i % 200 == 0:
            print("Generation Process: {:05.2f}%".format(i / n_slice * 100))
        x_tmp, _ = torchaudio.load(wav_file, normalization=True)
        j = random.randint(0, x_tmp.shape[1] - win_size)
        x[:, i * slice_size:(i + 1) * slice_size + x_fade // 2] += x_tmp[:, j:j + win_size] * tukey_win

    clip, sr = torchaudio.load(os.path.join("Data", "clip_101.aif"), normalization=True)
    x = torch.cat((clip[0:channels, :], x, clip[0:channels, :]), dim=1)

    return x


files = glob.glob(os.path.join('Song_Collection', '*.wav'))
x_train = dataset_generator(files=files, n_slice=100000, slice_size=4096+6000, x_fade=3000, channels=2)
torchaudio.save("Data/210317_Dataset.wav", x_train, 44100)

