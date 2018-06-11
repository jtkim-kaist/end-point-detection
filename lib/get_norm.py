import sys
sys.path.insert(0, './lib')
import os
import numpy as np
import glob
from lib import utils
import scipy.io.wavfile, scipy.signal, scipy.io
import librosa
from multiprocessing import Process, Queue
from lib import config
import matplotlib.pyplot as plt
import math
import numpy as np


def get_stft(file_list, output, mode='mfsc'):

    nperseg = config.nperseg

    item = []

    if mode is 'stft':

        mag_mean_list = []
        mag_std_list = []
        phase_mean_list = []
        phase_std_list = []

        for fname in file_list:
            data, rate = librosa.load(fname, config.fs)
            data = power_normalize(data)

            Zxx = librosa.core.stft(data, n_fft=config.nfft, hop_length=int(nperseg*0.5), win_length=int(nperseg))
            # Zxx = librosa.feature.melspectrogram(data, sr=rate, n_fft=int(nfft), hop_length=int(nperseg*0.5),
            #                                      n_mels=config.n_mels)
            # _, _, Zxx = scipy.signal.stft(data, fs=rate, nperseg=nperseg, nfft=nfft)
            mag_mean = np.mean(np.abs(Zxx), axis=1)
            mag_std = np.std(np.abs(Zxx), axis=1)
            phase_mean = np.mean(np.angle(Zxx), axis=1)
            phase_std = np.std(np.angle(Zxx), axis=1)

            mag_mean_list.append(mag_mean)
            mag_std_list.append(mag_std)

            phase_mean_list.append(phase_mean)
            phase_std_list.append(phase_std)

        mag_mean = np.mean(np.asarray(mag_mean_list), axis=0)
        mag_std = np.mean(np.asarray(mag_std_list), axis=0)
        phase_mean = np.mean(np.asarray(phase_mean_list), axis=0)
        phase_std = np.mean(np.asarray(phase_std_list), axis=0)

        item.append(mag_mean)
        item.append(mag_std)
        item.append(phase_mean)
        item.append(phase_std)

        output.put(item)

    elif mode is 'mfsc':

        mag_mean_list = []
        mag_std_list = []

        for fname in file_list:
            data, rate = librosa.load(fname, config.fs)
            # data = power_normalize(data)
            nfft = 2 ** (np.floor(np.log2(nperseg) + 1))
            # Zt = librosa.core.stft(data, n_fft=config.nfft, hop_length=int(nperseg*0.5), win_length=int(nperseg))
            Zxx = librosa.feature.melspectrogram(data, sr=rate, n_fft=int(nfft), hop_length=int(nperseg * 0.5),
                                                 n_mels=config.n_mels, fmin=300, fmax=8000)
            # _, _, Zxx = scipy.signal.stft(data, fs=rate, nperseg=nperseg, nfft=nfft)
            mag_mean = np.mean(np.abs(Zxx), axis=1)
            mag_std = np.std(np.abs(Zxx), axis=1)

            mag_mean_list.append(mag_mean)
            mag_std_list.append(mag_std)

        mag_mean = np.mean(np.asarray(mag_mean_list), axis=0)
        mag_std = np.mean(np.asarray(mag_std_list), axis=0)

        item.append(mag_mean)
        item.append(mag_std)

        output.put(item)


def fnctot(input_file_list, dist_num, mode='mfsc'):

    split_file_list = utils.chunkIt(input_file_list, dist_num)

    queue_list = []
    procs = []

    for i, file_list in enumerate(split_file_list):
        queue_list.append(Queue())  # define queues for saving the outputs of functions
        procs.append(Process(target=get_stft, args=(
            file_list, queue_list[i], mode)))  # define process

    for p in procs:  # process start
        p.start()

    M_list = []

    for i in range(dist_num):  # save results from queues and close queues
        M_list.append(queue_list[i].get())
        queue_list[i].close()

    for p in procs:  # close process
        p.join()

    return M_list


def power_normalize(sig):
    beta = 1000 / math.sqrt(np.sum(sig ** 2) / (sig.shape[0]))
    sig = sig * beta
    return sig


# def calc_norm():
#     mode = 'mfsc'
#     distribution_num = 4
#
#     win_len = config.win_len
#     nperseg = config.nperseg
#
#     train_path = os.path.abspath('../data/train/wav')
#     input_file_list = sorted(glob.glob(train_path + '/*.wav'))
#
#     result = fnctot(input_file_list, config.dist_num, mode)
#     result = np.mean(np.asarray(result), axis=0)
#
#     if mode is 'mfsc':
#         result = {'mag_mean': result[0], 'mag_std': result[1]}
#
#     elif mode is 'stft':
#         result = {'mag_mean': result[0], 'mag_std': result[1], 'phase_mean': result[2], 'phase_std': result[3]}
#
#     scipy.io.savemat('../data/train/norm/norm.mat', result)


if __name__ == '__main__':
    # calc_norm()
    mode = 'mfsc'
    distribution_num = 4

    win_len = config.win_len
    nperseg = config.nperseg

    train_path = os.path.abspath('../data/train/wav')
    input_file_list = sorted(glob.glob(train_path + '/*.wav'))

    result = fnctot(input_file_list, distribution_num, mode)
    result = np.mean(np.asarray(result), axis=0)

    if mode is 'mfsc':
        result = {'mag_mean': result[0], 'mag_std': result[1]}

    elif mode is 'stft':
        result = {'mag_mean': result[0], 'mag_std': result[1], 'phase_mean': result[2], 'phase_std': result[3]}

    scipy.io.savemat('../data/train/norm/norm.mat', result)
