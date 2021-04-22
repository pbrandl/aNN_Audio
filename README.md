# aNN_Audio

Modelling a digital twin of an analog distortion device for audio signals. Essentially, the device adds overtones to the signal. The current implementation uses a modified version of WaveNet.

The data used for training is randomly cut and concatenated from a MedleyDB -- a dataset of multitrack audio for music research. The obtained audio concatenation was then recorded by an analog harmonic distortion device, i.e., the input X is considered a raw audio file and differs from the target Y only by the distortion effect.

Due to computational complexity the project is mostly implemented in Google Colab.

## Receptive Field

The WaveNet is constructed to predict an arbitrary length of an audio file. In order to achieve that the input audio file is divided in n parts. Then, each divison <img src="https://render.githubusercontent.com/render/math?math=x_i \in [x_0, ... x_n]"> is predicted by a forward pass through the model. However, this leads to missing information about the audio signal, i.e., the previous audio signal of 

# Reference
- R. Bittner, J. Salamon, M. Tierney, M. Mauch, C. Cannam & J. P. Bello, "MedleyDB: A Multitrack Dataset for Annotation-Intensive MIR Research", in 15th International Society for Music Information Retrieval Conference, Taipei, Taiwan, 2014, https://medleydb.weebly.com/.
- A. van den Oord, S. Dieleman, H. Zen, K. Simonyan, O. Vinyals, A. Graves, N. Kalchbrenner, A. Senior & K. Kavukcuoglu, "WaveNet: A Generative Model for Raw Audio", in CoRR, 2016, http://arxiv.org/abs/1609.03499.
