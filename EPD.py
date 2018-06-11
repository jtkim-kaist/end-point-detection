import numpy as np
import matplotlib.pyplot as plt

vad_result = np.ones((1000, 1))
vad_result[1:100, :] = 0
vad_result[900:, :] = 0

vad_result[150:200, :] = 0
vad_result[850:860, :] = 0

vad_result[950:960, :] = 1
plt.subplot(2, 1, 1)
plt.plot(vad_result*0.5, 'g')
# plt.show()

speech_start = 0
speech_end = 0
speech_threshold = 0.5
hangover = 2
min_speech_len = 300

speech_buffer_size = 15
input_speech_buffer = np.zeros((speech_buffer_size, 1))
speech_indx = 0

speech_data = []

while True:

    # input_speech_buffer = vad_func(speech_seg)
    input_speech_buffer = vad_result[speech_indx:speech_indx+speech_buffer_size]
    xx = np.arange(speech_indx, speech_indx+speech_buffer_size)
    # plt.step(xx, input_speech_buffer, 'r')
    print(speech_start)
    if np.mean(input_speech_buffer) >= speech_threshold and speech_start == 0:
        # start speech detection
        speech_start = 1
        speech_data.append(speech_indx)
        speech_data.append(speech_indx + speech_buffer_size)

    elif np.mean(input_speech_buffer) >= speech_threshold and speech_start == 1:
        # hold the speech detection
        hangover = 2
        speech_start = 1
        speech_data.append(speech_indx)
        speech_data.append(speech_indx)

    elif np.mean(input_speech_buffer) <= speech_threshold and speech_start == 0:
        # do not start speech detection
        speech_start = 0
    elif np.mean(input_speech_buffer) <= speech_threshold and speech_start == 1:
        if hangover > 0:
            # wait which the speech detection is really ended or not
            hangover = hangover - 1
        else:
            speech_start = 0
            speech_data = np.asarray(speech_data)
            if speech_data[-1] - speech_data[0] >= min_speech_len:
                # if the length of detected speech is more than minimum length of speech, save the speech
                print("save speech")
                print(speech_data.shape[0])
                break
            else:
                # if the length of detected speech is less than minimum length of speech, holding the EPD
                print(speech_data.shape[0])
                speech_start = 0
                hangover = 2
                speech_data = []

    speech_indx = speech_indx + speech_buffer_size

print(speech_data[0])
print(speech_data[-1])
plt.subplot(2, 1, 2)
plt.plot(vad_result[speech_data[0]:speech_data[-1]])
plt.show()

aa = 1

# data buffer 만들어서 hang-before 처리하고, hang-over부분 수정해서 hangover 처리

