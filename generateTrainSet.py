import os
import glob
import random
import torch
import torchaudio

dir = r'Data_Collection'
sample_size = 18000
sample_num = 5000

train_files = glob.glob(os.path.join(dir, '*'))

x = torch.zeros(2, sample_size * sample_num)

for i in range(sample_num):
    print("Sampling {:04.2f}".format(i/sample_num))
    wav_file = random.choice(train_files)
    tmp_x, _ = torchaudio.load(wav_file, normalization=False)
    len_x = tmp_x.shape[1]
    j = random.randint(0, len_x - sample_size)
    x[:, i * sample_size:(i + 1) * sample_size] = tmp_x[:, j:j + sample_size]

clip, sr = torchaudio.load(os.path.join("Data", "clip.wav"), normalization=False)

print("Clip shape", clip.shape)
print("Data shape", x.shape)
x = torch.cat((clip, x, clip), dim=1)
print("ClipData shape", x.shape)

torchaudio.save("Data/rand_train_%i.wav" % sample_size, x, sr)
