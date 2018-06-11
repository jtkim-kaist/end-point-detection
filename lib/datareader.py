import numpy as np
import os
import glob
from lib import utils
import librosa
import time
from lib import config
import math
from multiprocessing import Process, Queue
import scipy.io, scipy.io.wavfile, scipy.signal
from sklearn.feature_extraction import image


class DataReader(object):

    def __init__(self, input_dir, output_dir, norm_dir, dist_num, mode='mfsc', is_training=True):

        self._is_training = is_training

        self._win_len = config.win_len
        self._nperseg = config.nperseg
        self._time_width = config.time_width
        self._dist_num = dist_num
        self._inpur_dir = input_dir
        self._output_dir = output_dir
        self._norm_dir = norm_dir
        self._mode = mode
        self._batch_size = 0
        if self._is_training:
            self._input_file_list = sorted(glob.glob(input_dir+'/*.wav'))
        else:
            self._input_file_list = [input_dir]  # wave directory

        self._output_file_list = sorted(glob.glob(output_dir+'/*.mat'))

        self._file_len = len(self._input_file_list)

        self._num_file = 0
        self._start_idx = 0

        self.eof = False
        self.file_change = False
        self.num_samples = 0

        self._inputs = 0

        if self._is_training:
            self._outputs = 0
            assert self._file_len == len(self._output_file_list)

        self._train_mean, self._train_std = self.norm_process(norm_dir+'/norm.mat')

    def norm_process(self, norm_dir):

        norm_param = scipy.io.loadmat(norm_dir)

        if self._mode is 'mfsc':
            mag_mean = norm_param["mag_mean"]
            mag_std = norm_param["mag_std"]

            train_mean = np.transpose(mag_mean, (1, 0))
            train_std = np.transpose(mag_std, (1, 0))

            # train_mean = np.transpose(np.concatenate((mag_mean, phase_mean), axis=0), (1, 0))
            # train_std = np.transpose(np.concatenate((mag_std, phase_std), axis=0), (1, 0))
        elif self._mode is 'stft':
            mag_mean = norm_param["mag_mean"]
            mag_std = norm_param["mag_std"]
            phase_mean = norm_param["phase_mean"]
            phase_std = norm_param["phase_std"]

            train_mean = np.transpose(np.concatenate((mag_mean, phase_mean), axis=0), (1, 0))
            train_std = np.transpose(np.concatenate((mag_std, phase_std), axis=0), (1, 0))
        return train_mean, train_std

    def next_batch(self, batch_size):

        self._batch_size = batch_size

        if self._start_idx == 0:

            self._inputs = self._read_input(self._input_file_list[self._num_file])

            # self._inputs = self._padding(self._read_input(self._input_file_list[self._num_file]),
            #                              batch_size, self._time_width)
            # self._inputs = self._normalize(self._inputs)
            # self._inputs = np.reshape(self._inputs, (-1, self._time_width,) + self._inputs.shape[1:])

            if self._is_training:

                self._outputs = self._read_output(self._output_file_list[self._num_file])

                # self._outputs = self._padding(self._read_output(self._output_file_list[self._num_file]),
                #                               batch_size, self._time_width)
                # self._outputs = np.reshape(self._outputs, (-1, self._time_width))

                assert np.shape(self._inputs)[0] == np.shape(self._outputs)[0], \
                    ("# samples is not matched between input: %d and output: %d files"
                     % (np.shape(self._inputs)[0], np.shape(self._outputs)[0]))

            # self.num_samples = np.shape(self._outputs)[0]
            self.num_samples = np.shape(self._inputs)[0]

        if self._start_idx + batch_size > self.num_samples:

            self._start_idx = 0
            self.file_change = True
            self._num_file += 1

            if self._num_file > self._file_len - 1:
                self.eof = True
                self._num_file = 0

            self._inputs = self._read_input(self._input_file_list[self._num_file])

            # self._inputs = self._padding(self._read_input(self._input_file_list[self._num_file]),
            #                              batch_size, self._time_width)
            # self._inputs = self._normalize(self._inputs)
            # self._inputs = np.reshape(self._inputs, (-1, self._time_width,) + self._inputs.shape[1:])

            if self._is_training:

                self._outputs = self._read_output(self._output_file_list[self._num_file])

                # self._outputs = self._padding(self._read_output(self._output_file_list[self._num_file]),
                #                               batch_size, self._time_width)
                # self._outputs = np.reshape(self._outputs, (-1, self._time_width))

                assert np.shape(self._inputs)[0] == np.shape(self._outputs)[0], \
                    ("# samples is not matched between input: %d and output: %d files"
                     % (np.shape(self._inputs)[0], np.shape(self._outputs)[0]))

            self.num_samples = np.shape(self._inputs)[0]

        else:
            self.file_change = False
            self.eof = False

        inputs = self._inputs[self._start_idx:self._start_idx + batch_size, :]

        if self._is_training:

            outputs = self._outputs[self._start_idx:self._start_idx + batch_size]
        else:
            outputs = np.zeros((inputs.shape[0]))

        self._start_idx += batch_size

        return inputs, outputs

    def _normalize(self, x):
        x = (x - self._train_mean)/self._train_std
        return x

    def _read_input(self, input_file_dir):
        
        dataname = os.path.dirname(input_file_dir) + '/' + os.path.basename(input_file_dir).split('.')[0] + '.npy'

        if self._is_training:
            if os.path.exists(dataname):
                feat = np.load(dataname)
            else:
                data, _ = librosa.load(input_file_dir, config.fs)
                # data = self._power_normalize(data)
                # _, data = scipy.io.wavfile.read(input_file_dir)
                # self._nperseg = rate*self._win_len

                feat= self.stft_dist(data, self._dist_num)

                np.save(dataname, feat)
        else:
            data, _ = librosa.load(input_file_dir, config.fs)
            # data = data/np.max(np.abs(data))

            # data = self._power_normalize(data)
            # _, data = scipy.io.wavfile.read(input_file_dir)
            # self._nperseg = rate*self._win_len

            feat = self.stft_dist(data, self._dist_num)

        return feat

    def _read_output(self, output_file_dir):
        label = np.squeeze(scipy.io.loadmat(output_file_dir)["label"])
        label = np.mean(librosa.util.frame(label, frame_length=int(self._nperseg), hop_length=int(self._nperseg*0.5)), axis=0)
        label = (label >= 0.5).choose(label, 1)
        label = (label < 0.5).choose(label, 0).astype(np.int32)
        print(label.shape)
        label = self._padding2(label, self._batch_size)
        return label

    @staticmethod
    def _power_normalize(sig):
        beta = 1000 / (math.sqrt(np.sum(sig ** 2)) / (sig.shape[0]))
        sig = sig * beta
        return sig

    @staticmethod
    def _padding(inputs, batch_size, width):
        pad_size = batch_size * width - inputs.shape[0] % (batch_size * width)
        pad_shape = (pad_size,) + inputs.shape[1:]
        inputs = np.concatenate((inputs, np.zeros(pad_shape, dtype=np.float32)))

        # window_pad = np.zeros((w_val, inputs.shape[1]))
        # inputs = np.concatenate((window_pad, inputs, window_pad), axis=0)
        return inputs

    @staticmethod
    def _padding2(inputs, batch_size):
        pad_size = batch_size - inputs.shape[0] % (batch_size)
        pad_shape = (pad_size,) + inputs.shape[1:]
        inputs = np.concatenate((inputs, np.zeros(pad_shape, dtype=np.float32)))

        # window_pad = np.zeros((w_val, inputs.shape[1]))
        # inputs = np.concatenate((window_pad, inputs, window_pad), axis=0)
        return inputs

    def stft_dist(self, data, dist_num):

        data = data.tolist()
        data_list = utils.chunkIt(data, dist_num)

        queue_list = []
        procs = []

        for i, data in enumerate(data_list):
            queue_list.append(Queue())  # define queues for saving the outputs of functions
            procs.append(Process(target=self.get_stft, args=(
                data, queue_list[i])))  # define process

        for p in procs:  # process start
            p.start()

        M_list = []

        for i in range(dist_num):  # save results from queues and close queues
            M_list.append(queue_list[i].get())
            queue_list[i].close()

        for p in procs:  # close process
            p.join()

        result = np.asarray(M_list)
        print(result.shape)
        result = np.reshape(result, (-1, result.shape[2], result.shape[3]))

        pad = np.expand_dims(np.zeros((int(config.time_width/2), result.shape[1])), axis=2)

        result = self._normalize(result)
        print(result.shape)
        result = np.squeeze(np.concatenate((pad, result, pad), axis=0))
        print(result.shape)
        result = image.extract_patches_2d(result, (config.time_width, config.n_mels))
        print(result.shape)
        result = np.expand_dims(self._padding2(result, self._batch_size), axis=3)
        # print(result.shape)
        return result

    def get_stft(self, data, output):

        data = np.asarray(data).astype(dtype=np.float32)
        # nfft = np.int(2**(np.floor(np.log2(self._nperseg)+1)))

        # _, _, Zxx = scipy.signal.stft(data, fs=fs, nperseg=self._nperseg, nfft=int(nfft))
        if self._mode is 'mfsc':
            Zxx = librosa.feature.melspectrogram(data, sr=config.fs, n_fft=int(config.nfft),
                                                 hop_length=int(self._nperseg * 0.5),
                                                 n_mels=config.n_mels, fmin=300, fmax=8000)

            mfsc = np.transpose(np.expand_dims(Zxx, axis=2), (1, 0, 2))[:-1, :]

            # label = np.mean(
            #     librosa.util.frame(data, frame_length=int(self._nperseg), hop_length=int(self._nperseg * 0.5)), axis=0)

            output.put(mfsc)

        elif self._mode is 'stft':

            Zxx = librosa.core.stft(data, n_fft=config.nfft, hop_length=int(self._nperseg*0.5), win_length=int(self._nperseg))

            mag = np.expand_dims(np.abs(Zxx), axis=2)
            phase = np.expand_dims(np.angle(Zxx), axis=2)
            stft = np.transpose(np.concatenate((mag, phase), axis=2), (1, 0, 2))[:-1, :]

            output.put(stft)

    def reader_initialize(self):
        self._num_file = 0
        self._start_idx = 0
        self.eof = False

    def eof_checker(self):
        return self.eof

    def file_change_checker(self):
        return self.file_change

    def file_change_initialize(self):
        self.file_change = False

if __name__ == '__main__':
    dist_num = 4
    mode = 'mfsc'
    train_input_path = os.path.abspath('../data/train/wav')
    train_output_path = os.path.abspath('../data/train/lab')
    norm_path = os.path.abspath('../data/train/norm')

    train_dr = DataReader(train_input_path, train_output_path, norm_path, dist_num, mode, is_training=True)
    while True:
        sample = train_dr.next_batch(512)
        print(train_dr._num_file)


    a = 1