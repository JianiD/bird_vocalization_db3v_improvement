import librosa
import librosa.display
import os, sys, cv2, matplotlib.pyplot as plt, numpy as np, shutil, itertools, pickle, pandas as pd, seaborn as sn, math, time
from random import seed, random, randint
import random
from sklearn.preprocessing import normalize
import soundfile as sf
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split



RANDOM_SEED = 830
SAMPLE_RATE = 16000
SIGNAL_LENGTH = 8  # seconds
FMIN = 20
FMAX = 8000

# Read metadata file
metadata = pd.read_csv(os.getcwd()+"\\single\\cross_S1train.csv")
header = list(metadata.head())

# Get bird names
bird_name = metadata['primary_label'].values
u, f = np.unique(bird_name, return_counts=True)

#print(u)

uniq_birds = list(u)
print(uniq_birds)
data_train = []
data_test = []
y_train = []
y_test = []
bird_name_dict = {}

# Get file_id corresponding to bird names
for i in range(len(uniq_birds)) :
    df = metadata[metadata['primary_label'] == uniq_birds[i]]
    df = df['filename'].values
    df = df.tolist()
    data_train.append(df[0])
    y_train.append(i)
    bird_name_dict[i] = uniq_birds[i]
    data_test += df[0:]
    y_test += [i] * (len(df) - 0)

data_train = data_test
y_train = y_test
print(len(y_train))

# Read training data and split into frames
frames_train = []
frames_test = []
y_frames_train = []
y_frames_test = []

features_log_mel = []

sum = 0
for i in tqdm(range(len(data_train))):
    # Read audio 读取音频
    curr = data_train[i]
    curr = "E:/data/cross/S1/S1val/"+ str(curr)
    if os.path.exists(curr):
        sig, rate = librosa.load(curr,sr=SAMPLE_RATE,mono=True, offset=None, duration=8)

        sig_splits = []
        for j in range(0, len(sig), int(SIGNAL_LENGTH * SAMPLE_RATE)):
            split = sig[j:j + int(SIGNAL_LENGTH * SAMPLE_RATE)]

            # End of signal?
            if len(split) < int(SIGNAL_LENGTH * SAMPLE_RATE):
                break

            sig_splits.append(split)

        # Extract mel spectrograms for each audio chunk
        s_cnt = 0
        saved_samples = []
        for chunk in sig_splits:
            # hop_length = int(SIGNAL_LENGTH * SAMPLE_RATE / (SPEC_SHAPE[1] - 1))
            hop_length = 512
            mel_spec = librosa.feature.melspectrogram(y=chunk,
                                                      sr=SAMPLE_RATE,
                                                      n_fft=2048,
                                                      hop_length=hop_length,
                                                      # n_mels=SPEC_SHAPE[0],
                                                      n_mels=128,
                                                      fmin=FMIN,
                                                      fmax=FMAX)
            # mel_spec = librosa.pcen(mel_spec)  #pcen
            mel_spec = librosa.power_to_db(mel_spec, ref=np.max)  #log-Mel

            # Normalize
            mel_spec -= mel_spec.min()
            mel_spec /= mel_spec.max()
            # print(mel_spec.shape)

            frames_train.append(mel_spec)  # frames_train is a list
            # try:
            #     print(y_train[i])
            # except:
            #     print('wrong data：',i)
            y_frames_train.append(y_train[i])

y_frames_train = np.array(y_frames_train)
# print(y_frames_train.shape)
# print(frames_train[0].shape)
r,c = frames_train[0].shape
frames_train = np.array(frames_train)
frames_train = frames_train.reshape((len(frames_train), r, c))

print('train_num:',len(y_frames_train))
# print('test_num:',len(y_frames_test))

# Write training and testing data into a pickle file
f = open(os.getcwd() + "/cross_S1train.pkl", 'wb')
pickle.dump([frames_train, y_frames_train], f)
f.close()
