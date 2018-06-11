import vad_func
import os
import matplotlib.pyplot as plt
import librosa
import numpy as np
import time
from lib import config


prj_dir = os.path.dirname(__file__)
wav_dir = os.path.abspath(prj_dir + '/data/test/dddd.wav')
data, _ = librosa.load(wav_dir, config.fs)


def epd(speech):

    fs = 16000.0
    speech_seg = int(fs * 0.001 * 100)  # 100 ms
    speech_minlen = int(fs * 0.001 * 2000)  # 1000 ms
    speech_maxlen = int(fs * 0.001 * 5000)  # 5000 ms

    speech_threshold = 0.5
    transition_buffer_size = 2
    transition_buffer = transition_buffer_size

    data_queue_size = 2
    speech_indx = 0
    speech_state = 0

    speech_data = []
    data_queue = []
    speech_queue = []

    while True:

        # get speech data stream

        if speech_indx + speech_seg < speech.shape[0]:
            input_speech_buffer = speech[speech_indx:speech_indx+speech_seg]
            no_speech = 0
        else:
            input_speech_buffer = speech[speech_indx:]
            no_speech = 1

        # save data into the queue
        if len(data_queue) < data_queue_size:
            data_queue.append(input_speech_buffer)
        elif len(data_queue) == data_queue_size:
            data_queue.pop(0)
            data_queue.append(input_speech_buffer)
        else:
            assert False, "queue size error"

        # conduct vad
        vad_result = vad_func.vad(input_speech_buffer)

        if np.mean(vad_result) >= speech_threshold and speech_state == 0:
            # start speech detection
            speech_state = 1
            speech_data.append(speech_indx)
            speech_data.append(speech_indx + speech_seg)
            speech_queue.append(np.concatenate(data_queue, axis=0))

        elif np.mean(vad_result) >= speech_threshold and speech_state == 1:
            # hold the speech detection
            transition_buffer = transition_buffer_size
            speech_state = 1
            speech_data.append(speech_indx)
            speech_data.append(speech_indx + speech_seg)
            speech_queue.append(input_speech_buffer)
        elif np.mean(vad_result) < speech_threshold and speech_state == 0:
            # do not start speech detection
            speech_state = 0
            speech_queue = []
        elif np.mean(vad_result) < speech_threshold and speech_state == 1:
            if transition_buffer > 0:
                # wait which the speech detection is really ended or not
                transition_buffer = transition_buffer - 1
                speech_data.append(speech_indx)
                speech_data.append(speech_indx + speech_seg)
                speech_queue.append(input_speech_buffer)
            else:
                speech_data = np.asarray(speech_data)
                if speech_data[-1] - speech_data[0] >= speech_minlen:
                    # if the length of detected speech is more than minimum length of speech, save the speech
                    speech_final = np.concatenate(speech_queue, axis=0)
                    print("save speech")
                    librosa.output.write_wav(prj_dir + '/epd.wav', speech_final, config.fs)
                    break
                elif speech_data[-1] - speech_data[0] >= speech_maxlen:
                    print("exceed the maximum speech length")
                    print("save speech")
                    librosa.output.write_wav(prj_dir + '/epd.wav', speech_final, config.fs)
                    break
                else:
                    # if the length of detected speech is less than minimum length of speech, holding the EPD
                    print(speech_data.shape[0])
                    speech_state = 0
                    transition_buffer = transition_buffer_size
                    speech_data = []
                    speech_queue = []

        speech_indx = speech_indx + speech_seg

        if no_speech:
            break


if __name__ == '__main__':
    epd(data)
