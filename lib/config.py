import numpy as np
import os

fs = 16000

time_width = int(11)
win_len = 0.020
nperseg = int(fs * win_len)
nfft = np.int(2 ** (np.floor(np.log2(nperseg) + 1)))
n_mels = int(80)

# lr = 0.0000005
# lr = 0.000003 # for mode 1
lr = 0.0001

device = '/gpu:0'

# logs_dir = os.path.abspath('../logs')

dist_num = int(8)

max_epoch = int(1350*10)

batch_size = int(1024)

val_step = int(450)

cell_size = int(32)

test_batch_size = int(20)

'''epd configuration'''

win_size = int(fs * 0.001 * 200)  # 200 ms
speech_minlen = 2000
speech_maxlen = 10000
speech_threshold = 0.5
transition_buffer_size = 2
save_dir = './samples'
speech_enhancement = True
