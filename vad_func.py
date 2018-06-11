import os
from lib import graph_test as gt
from lib import utils
from lib import config
from sklearn.feature_extraction import image
import scipy.io, scipy.io.wavfile, scipy.signal
import librosa
import glob
import tensorflow as tf
import numpy as np
import time

prj_dir = os.path.dirname(__file__)
norm_dir = os.path.abspath(prj_dir + '/data/train/norm'+'/norm.mat')
graph_name = sorted(glob.glob(prj_dir + '/saved_model/*.pb'))[-1]
graph = gt.load_graph(graph_name)


norm_param = scipy.io.loadmat(norm_dir)

mag_mean = norm_param["mag_mean"]
mag_std = norm_param["mag_std"]

train_mean = np.transpose(mag_mean, (1, 0))
train_std = np.transpose(mag_std, (1, 0))

speech_pad = np.zeros(int(config.fs * 0.001 * 50))


def get_feature(data):

    result = get_mfsc(data)

    result = (result - train_mean) / train_std
    pad = np.expand_dims(np.zeros((int(config.time_width / 2), result.shape[1])), axis=2)

    result = np.squeeze(np.concatenate((pad, result, pad), axis=0))

    result = image.extract_patches_2d(result, (config.time_width, config.n_mels))

    result = np.expand_dims(result, axis=3)
    return result


def get_mfsc(data):

    data = np.asarray(data).astype(dtype=np.float32)

    Zxx = librosa.feature.melspectrogram(data, sr=config.fs, n_fft=int(config.nfft),
                                         hop_length=int(config.nperseg * 0.5),
                                         n_mels=config.n_mels, fmin=300, fmax=8000)

    mfsc = np.transpose(np.expand_dims(Zxx, axis=2), (1, 0, 2))[:-1, :]

    return mfsc


def vad(speech):

    speech = np.concatenate([speech_pad, speech, speech_pad], axis=0)

    node_inputs = graph.get_tensor_by_name('prefix/model_1/inputs:0')
    node_labels = graph.get_tensor_by_name('prefix/model_1/labels:0')
    node_softpred = graph.get_tensor_by_name('prefix/model_1/softpred:0')

    inputs = get_feature(speech)
    fake_outputs = np.zeros(inputs.shape[0])
    feed_dict = {node_inputs: inputs, node_labels: fake_outputs}
    start_time = time.time()

    sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    # sess_config.gpu_options.allow_growth = True
    with tf.Session(graph=graph, config=sess_config) as sess:
        softpred = sess.run(node_softpred, feed_dict=feed_dict)

    softpred = softpred[5:-5]
    pred = np.int32(softpred > 0.7)
    raw_pred = utils.frame2rawlabel(pred, config.nperseg, int(config.nperseg * 0.5))[0:speech.shape[0]]

    print(time.time() - start_time)

    return raw_pred


if __name__ == '__main__':
    pass
