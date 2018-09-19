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


norm_dir = os.path.abspath('./data/train/norm'+'/norm.mat')
graph_name = sorted(glob.glob(os.path.abspath('./saved_model2/*.pb')))[-1]
graph = gt.load_graph(graph_name)
sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
sess = tf.Session(graph=graph, config=sess_config)


def get_feature(data):

    data = np.asarray(data).astype(dtype=np.float32)

    Zxx = librosa.feature.melspectrogram(data, sr=config.fs, n_fft=int(config.nfft),
                                         hop_length=int(config.nperseg * 0.5),
                                         n_mels=config.n_mels, fmin=300, fmax=4000)  # (freq, frame number)
    Zxx = librosa.core.power_to_db(Zxx)

    mfsc = np.transpose(np.expand_dims(Zxx, axis=2), (1, 0, 2))[:-1, :]

    pad = np.expand_dims(np.zeros((int(config.time_width / 2), mfsc.shape[1])), axis=2)

    mfsc = np.squeeze(np.concatenate((pad, mfsc, pad), axis=0))
    # print(result.shape)
    mfsc = image.extract_patches_2d(mfsc, (config.time_width, config.n_mels))
    # print(result.shape)
    # result = np.expand_dims(self._padding2(result, self._batch_size), axis=3)
    mfsc = np.expand_dims(mfsc, axis=3)
    mfsc = local_norm(mfsc)

    return mfsc


def local_norm(x):
    x = np.transpose(np.squeeze(x), [1, 0, 2])
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    x = (x - mean) / (std + 1e-3)

    x = np.expand_dims(np.transpose(x, [1, 0, 2]), axis=3)

    return x


def vad(speech):

    node_inputs = graph.get_tensor_by_name('prefix/model_1/inputs:0')
    node_labels = graph.get_tensor_by_name('prefix/model_1/labels:0')
    node_prediction = graph.get_tensor_by_name('prefix/model_1/prediction:0')
    node_softpred = graph.get_tensor_by_name('prefix/model_1/softpred:0')

    inputs = get_feature(speech)
    fake_outputs = np.zeros(inputs.shape[0])

    feed_dict = {node_inputs: inputs, node_labels: fake_outputs}

    pred, _, softpred = sess.run([node_prediction, node_labels, node_softpred], feed_dict=feed_dict)
    raw_pred = utils.frame2rawlabel(pred, config.nperseg, int(config.nperseg * 0.5))[0:speech.shape[0]]

    return raw_pred




