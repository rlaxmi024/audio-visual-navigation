import glob
import random

import scipy
import torch
import pickle
import librosa
import skimage
import numpy as np
from scipy.io import wavfile


class SSLAudioDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, dataset, rir_sr, mode):
        self.sound_files = glob.glob(f'{dataset_dir}/sounds/1s_all/*')
        self.metadata_dir = f'{dataset_dir}/metadata/{dataset}'
        self.env_dirs = glob.glob(f'{dataset_dir}/binaural_rirs/{dataset}/*')
        random.shuffle(self.env_dirs)

        train_samples = 18
        if mode == 'train':
            self.env_dirs = self.env_dirs[: train_samples]
        elif mode == 'val':
            self.env_dirs = self.env_dirs[train_samples: ]

        self.n_fft = 512
        self.rir_sr = rir_sr
        self.hop_length = 160
        self.win_length = 400

        self.angle_dict = {0: 0, 90: 1, 180: 2, 270: 3}

    def __len__(self):
        return 10000

    def compute_stft(self, signal):
        stft = np.abs(librosa.stft(signal, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length))
        return skimage.measure.block_reduce(stft, block_size=(4, 4), func=np.mean)

    def _compute_spectrogram(self, audio_data):
        channel1_magnitude = np.log1p(self.compute_stft(audio_data[0]))
        channel2_magnitude = np.log1p(self.compute_stft(audio_data[1]))
        return np.stack([channel1_magnitude, channel2_magnitude], axis=-1)

    def _compute_euclidean_distance(self, graph, index):
        p1 = graph.nodes[int(index[0])]['point']
        p2 = graph.nodes[int(index[1])]['point']
        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[2] - p2[2]) ** 2)

    def __getitem__(self, _):
        sound_1 = random.sample(self.sound_files, 1)[0]
        if random.random() < .5:
            sound_2 = sound_1
            contrastive_label = 0
        else:
            sound_2 = random.sample([i for i in self.sound_files if i != sound_1], 1)[0]
            contrastive_label = 1
        
        env = random.sample(self.env_dirs, 1)[0]
        with open(f"{self.metadata_dir}/{env.rsplit('/',1)[1]}/graph.pkl", 'rb') as f:
            graph = pickle.load(f)

        rir_1 = random.sample(glob.glob(f'{env}/*/*'), 1)[0]
        source = rir_1.rsplit('/', 1)[1][: -4].rsplit('_')[1]
        rir_2 = random.sample(glob.glob(f'{env}/*/*_{source}.wav'), 1)[0]

        sound_1 = librosa.load(sound_1, sr=self.rir_sr)[0]
        sound_2 = librosa.load(sound_2, sr=self.rir_sr)[0]

        binaural_rir_1 = wavfile.read(rir_1)[1]
        binaural_rir_2 = wavfile.read(rir_2)[1]

        binaural_convolved_1 = np.array([scipy.signal.fftconvolve(sound_1, binaural_rir_1[:, i]) for i in [0, 1]])[:, : self.rir_sr]
        binaural_convolved_2 = np.array([scipy.signal.fftconvolve(sound_2, binaural_rir_2[:, i]) for i in [0, 1]])[:, : self.rir_sr]

        spec_1 = self._compute_spectrogram(binaural_convolved_1)
        spec_2 = self._compute_spectrogram(binaural_convolved_2)

        angle_1 = int(rir_1.rsplit('/',2)[1])
        angle_2 = int(rir_2.rsplit('/',2)[1])
        angle_label = angle_1 - angle_2
        angle_label = angle_label if angle_label >= 0 else 360 + angle_label
        angle_label = self.angle_dict[angle_label]

        dist_1 = self._compute_euclidean_distance(graph, rir_1.rsplit('/', 1)[1][: -4].rsplit('_'))
        dist_2 = self._compute_euclidean_distance(graph, rir_2.rsplit('/', 1)[1][: -4].rsplit('_'))
        eu_dist_label = 0.0 if dist_1 < dist_2 else 1.0

        return {
            'spec_1': spec_1,
            'spec_2': spec_2,
            'contrastive_label': contrastive_label,
            'angle_label': angle_label,
            'eu_dist_label': eu_dist_label
        }


# ssl_audio_dataset = torch.utils.data.DataLoader(
#     SSLAudioDataset('../../sound-spaces/data', 'replica', 44100),
#     batch_size=4,
#     shuffle=True,
#     num_workers=0
# )