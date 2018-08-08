import sys
sys.path.insert(0, './lib')

import vad_func
import os
import matplotlib.pyplot as plt
import librosa
import numpy as np
import time
import tensorflow as tf
from lib import config
import glob
import soundfile as sf

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""


class EPD(object):

    def __init__(self, win_size):

        self.fs = int(16000)
        self.speech_seg = win_size  # 100 ms
        self.speech_minlen = int(self.fs * 0.001 * 2000)  # 1000 ms
        self.speech_maxlen = int(self.fs * 0.001 * 10000)  # 5000 ms

        self.speech_threshold = 0.5

        self.transition_buffer_size = 5
        self.transition_buffer = self.transition_buffer_size

        self.data_queue_size = 2
        self.speech_indx = 0
        self.speech_state = 0

        self.speech_data = []
        self.data_queue = []
        self.speech_queue = []

        self.data_num = 0

    def epd(self, data_seg):

        input_speech_buffer = data_seg

        # save data into the queue

        if len(self.data_queue) < self.data_queue_size:
            self.data_queue.append(input_speech_buffer)
        elif len(self.data_queue) == self.data_queue_size:
            self.data_queue.pop(0)
            self.data_queue.append(input_speech_buffer)
        else:
            assert False, "queue size error"

        # conduct vad

        vad_result = vad_func.vad(input_speech_buffer)

        if np.mean(vad_result) >= self.speech_threshold and self.speech_state == 0:
            # start speech detection
            self.speech_state = 1
            self.speech_data.append(self.speech_indx)
            self.speech_data.append(self.speech_indx + self.speech_seg)
            self.speech_queue.append(np.concatenate(self.data_queue, axis=0))

        elif np.mean(vad_result) >= self.speech_threshold and self.speech_state == 1:
            # hold the speech detection
            self.transition_buffer = self.transition_buffer_size
            self.speech_state = 1
            self.speech_data.append(self.speech_indx)
            self.speech_data.append(self.speech_indx + self.speech_seg)
            self.speech_queue.append(input_speech_buffer)

        elif np.mean(vad_result) < self.speech_threshold and self.speech_state == 0:
            # do not start speech detection
            self.speech_state = 0
            self.speech_queue = []

        elif np.mean(vad_result) < self.speech_threshold and self.speech_state == 1:
            if self.transition_buffer > 0:
                # wait which the speech detection is really ended or not
                self.transition_buffer = self.transition_buffer - 1
                self.speech_data.append(self.speech_indx)
                self.speech_data.append(self.speech_indx + self.speech_seg)
                self.speech_queue.append(input_speech_buffer)
            else:
                self.speech_data = np.asarray(self.speech_data)
                if self.speech_data[-1] - self.speech_data[0] >= self.speech_minlen:
                    # if the length of detected speech is more than minimum length of speech, save the speech
                    speech_final = np.concatenate(self.speech_queue, axis=0)

                    print("save speech %d" % self.data_num)
                    librosa.output.write_wav(
                        './samples/epd_' + str(self.data_num).zfill(5) + '.wav', speech_final, self.fs)
                    self.data_num = self.data_num + 1

                    self.initialize()

                elif self.speech_data[-1] - self.speech_data[0] >= self.speech_maxlen:
                    speech_final = np.concatenate(self.speech_queue, axis=0)
                    print("exceed the maximum speech length")
                    print("save speech %d" % self.data_num)
                    librosa.output.write_wav(
                        './samples/epd_' + str(self.data_num).zfill(5) + '.wav', speech_final, self.fs)
                    self.data_num = self.data_num + 1
                    self.initialize()

                else:
                    # if the length of detected speech is less than minimum length of speech, holding the EPD
                    # print(speech_data.shape[0])
                    self.speech_state = 0
                    self.transition_buffer = self.transition_buffer_size
                    self.speech_data = []
                    self.speech_queue = []

        self.speech_indx = self.speech_indx + self.speech_seg

    def initialize(self):

        self.transition_buffer_size = 5
        self.transition_buffer = self.transition_buffer_size

        self.speech_indx = 0
        self.speech_state = 0

        self.speech_data = []
        self.data_queue = []
        self.speech_queue = []


if __name__ == '__main__':

    clean_list = sorted(glob.glob('./*.wav'))

    win_size = int(config.fs * 0.001 * 200)  # 200 ms

    data, sr = sf.read(clean_list[0])
    data = librosa.core.resample(data, sr, config.fs)
    data = np.concatenate([data, np.squeeze(np.random.random([30000, 1]) * 0.0001)], axis=0)
    print("total speech time: %.4f" % (data.shape[0]/config.fs))

    Detector = EPD(win_size)

    start_idx = 0
    start_time = time.time()

    while True:

        if start_idx + win_size > data.shape[0]:
            break
        data_stream = data[start_idx:start_idx+win_size]
        Detector.epd(data_stream)
        start_idx = start_idx + win_size

    end_time = time.time() - start_time
    print(end_time)
