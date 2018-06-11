import numpy as np
import os

time_width = int(7)
fs = 16000
win_len = 0.020
nperseg = int(fs * win_len)
nfft = np.int(2 ** (np.floor(np.log2(nperseg) + 1)))
n_mels = int(64)

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
