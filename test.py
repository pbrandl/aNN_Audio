import numpy as np
from scipy.io import wavfile

infile_1 = 'Audio/kick_pro_32_1.wav'
infile_2 = 'Audio/kick_pro_32_2.wav'

sr, data_1 = wavfile.read(infile_1)
_, data_2 = wavfile.read(infile_2)

data_res = data_1 - data_2

print(data_res)

wavfile.write('Output/subtraction.wav', sr, data_res)
