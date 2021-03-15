# aNN_Audio

Modelling a digital twin of an analog distortion device for audio signals. Essentially, the device adds overtones to the signal. The current implementation uses a modified version of WaveNet.

The data used for training is randomly cut and concatenated from a MedleyDB -- a dataset of multitrack audio for music research. The obtained audio concatenation was then recorded by an analog harmonic distortion device.

