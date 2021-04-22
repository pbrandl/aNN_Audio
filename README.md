# aNN_Audio

Modelling a digital twin of an analog distortion device for audio signals as shown in the image below. <img src="https://render.githubusercontent.com/render/math?math=x, y"> are audio signals, i.e., 16-bit wav files.
Essentially, the device adds overtones to the signal. The current implementation uses a modified version of WaveNet [2].

<p align="center">
  <img src="https://github.com/pbrandl/aNN_Audio/blob/master/images/concept.png?raw=true" width="50%" height="50%" alt="Conceptual Digital Twin" align="center">
</p>


Due to computational complexity of the project is mostly implemented in as Python notebook in Google Colab (`WaveNet.ipynb`). Due to the large training data set (currently 4 GB) the data is stored in Google Drive. 

## Training Data Generation

Training data is generated in `generateTrainSet.py`. The data used for training is randomly cut and concatenated from MedleyDB -- a dataset of multitrack audio for music research [1]. The obtained audio concatenation was then recorded through an analog harmonic distortion device, i.e., the input <img src="https://render.githubusercontent.com/render/math?math=x"> is considered a wav audio file and differs from the target file <img src="https://render.githubusercontent.com/render/math?math=y"> only by the distortion effect. (Note that the files are starting and ending with a loud click sound.
This helps for synchronization of the files after recording. This is done in `preprocessing.py`.


## Predicting an Audio Sequence

The WaveNet is constructed to predict an arbitrary length of an audio file. In order to achieve that the input audio file is divided in <img src="https://render.githubusercontent.com/render/math?math=n"> parts. Then, each divison <img src="https://render.githubusercontent.com/render/math?math=x_i \in [x_0, ... x_n]"> is predicted by a forward pass through the model. However, this leads to missing information about the preceeding audio signal <img src="https://render.githubusercontent.com/render/math?math=x_{i-1}">. Therefore, the ending of the signal <img src="https://render.githubusercontent.com/render/math?math=x_{i-1}"> is added to <img src="https://render.githubusercontent.com/render/math?math=x_{i}">. The size of the ending is defined by the receptive field length.

# Reference
- R. Bittner, J. Salamon, M. Tierney, M. Mauch, C. Cannam & J. P. Bello, "MedleyDB: A Multitrack Dataset for Annotation-Intensive MIR Research", in 15th International Society for Music Information Retrieval Conference, Taipei, Taiwan, 2014, https://medleydb.weebly.com/.
- A. van den Oord, S. Dieleman, H. Zen, K. Simonyan, O. Vinyals, A. Graves, N. Kalchbrenner, A. Senior & K. Kavukcuoglu, "WaveNet: A Generative Model for Raw Audio", in CoRR, 2016, http://arxiv.org/abs/1609.03499.
