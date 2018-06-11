import numpy as np
import os

time_width = int(17)
fs = 16000
win_len = 0.020
nperseg = int(fs * win_len)
nfft = np.int(2 ** (np.floor(np.log2(nperseg) + 1)))
n_mels = int(64)

# lr = 0.0000005
# lr = 0.000003 # for mode 1
lr = 0.0001

device = '/gpu:1'

# logs_dir = os.path.abspath('../logs')

dist_num = int(8)

max_epoch = int(160001)

batch_size = int(4024)
test_batch_size = int(4024)

val_step = int(5000)

cell_size = int(32)
