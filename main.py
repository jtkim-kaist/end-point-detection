import sys
import os
from lib import datareader as dr
from lib import train as tr
from lib import graph_save as gs
from lib import graph_test as gt
import glob
from lib import config
import scipy.io
import tensorflow as tf
import numpy as np
from lib import utils
import matplotlib.pyplot as plt
import time
import librosa
from lib import get_norm as gn

# train_path = os.path.abspath('./data/train')
# valid_path = os.path.abspath('./data/valid')
#
#
# dr(train_path, output_dir, norm_dir, sample_rate)


def vad_test(wave_dir):

    prj_dir = os.path.dirname(__file__)
    graph_name = sorted(glob.glob(prj_dir + '/saved_model/*.pb'))[-1]
    graph = gt.load_graph(graph_name)
    norm_path = os.path.abspath(prj_dir + '/data/train/norm')

    test_dr = dr.DataReader(wave_dir, '', norm_path, dist_num=config.dist_num, is_training=False)

    node_inputs = graph.get_tensor_by_name('prefix/model_1/inputs:0')
    node_labels = graph.get_tensor_by_name('prefix/model_1/labels:0')
    node_prediction = graph.get_tensor_by_name('prefix/model_1/prediction:0')
    node_softpred = graph.get_tensor_by_name('prefix/model_1/softpred:0')

    pred = []
    lab = []
    softpred =[]

    while True:

        test_inputs, test_labels = test_dr.next_batch(config.test_batch_size)

        feed_dict = {node_inputs: test_inputs, node_labels: test_labels}

        if test_dr. file_change_checker():
            pred = np.reshape(np.asarray(pred), [-1, 1])
            lab = np.reshape(np.asarray(lab), [-1, 1])
            softpred = np.reshape(np.asarray(softpred), [-1, 1])
            test_dr.reader_initialize()
            break

        sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess_config.gpu_options.allow_growth = True

        with tf.Session(graph=graph, config=sess_config) as sess:
            pred_temp, lab_temp, softpred_temp = sess.run([node_prediction, node_labels, node_softpred],
                                                          feed_dict=feed_dict)
        plt.plot(pred_temp)
        plt.show()
        pred.append(pred_temp)
        lab.append(lab_temp)
        softpred.append(softpred_temp)

    return pred, lab, softpred


if __name__ == '__main__':

    reset = True
    test_only = False
    train_reset = False
    logs_dir = os.path.abspath('./logs')
    save_dir = os.path.abspath('./saved_model')
    prj_dir = os.path.abspath('.')

    if reset:
        os.popen('rm -rf ' + logs_dir + '/*')
        os.popen('mkdir ' + logs_dir + '/train')
        os.popen('mkdir ' + logs_dir + '/valid')

    if train_reset:
        os.popen('rm -rf ' + './data/train/wav/*.npy')
        os.popen('rm -rf ' + './data/valid/wav/*.npy')

    # model train

    if not test_only:
        tr.main(prj_dir)

        # save graph

        gs.freeze_graph(logs_dir, save_dir, 'model_1/prediction,model_1/labels,model_1/softpred')

    # test graph

    # wav_dir = os.path.abspath('./data/test/concat.wav')
    # data, _ = librosa.load(wav_dir, config.fs)
    # data = data/np.max(np.abs(data))
    # t_axis = np.arange(0, data.shape[0]) / config.fs
    # start_time = time.time()
    # pred, _, softpred = vad_test(wav_dir)
    # elapsed_time = time.time() - start_time
    # print("%.4f ms/s" % (elapsed_time/t_axis[-1]*1000))
    # # plt.plot(softpred)
    # # plt.show()
    # raw_pred = utils.frame2rawlabel(pred, config.nperseg, int(config.nperseg*0.5))[0:data.shape[0]]
    #
    # f, axarr = plt.subplots(2)
    #
    # axarr[0].set_ylim([-1.2, 1.2])
    # axarr[0].set_xlim([0, t_axis[-1]])
    #
    # # axes.set_ylim([0, 2])
    #
    # axarr[0].plot(t_axis, raw_pred)
    # axarr[0].plot(t_axis, data)
    #
    # # plt.plot(softpred)
    # # plt.subplot(211)
    #
    # wav_dir = os.path.abspath('./data/test/clean_speech.wav')
    # data, _ = librosa.load(wav_dir, config.fs)
    # data = data/np.max(np.abs(data))
    # t_axis = np.arange(0, data.shape[0]) / config.fs
    #
    # pred, _, softpred = vad_test(wav_dir)
    #
    # raw_pred = utils.frame2rawlabel(pred, config.nperseg, int(config.nperseg*0.5))[0:data.shape[0]]
    #
    # axarr[1].set_ylim([-1.2, 1.2])
    # axarr[1].set_xlim([0, t_axis[-1]])
    #
    # axarr[1].plot(t_axis, raw_pred)
    # axarr[1].plot(t_axis, data)
    # # plt.subplot(212)
    # # axes = plt.gca()
    # # axes.set_ylim([0, 2])
    #
    # plt.show()
    #
    # wav_dir = os.path.abspath('./SNR_10_noise_white.wav')
    #
    # pred, _, softpred = vad_test(wav_dir)
    #
    # raw_pred = utils.frame2rawlabel(pred, config.nperseg, int(config.nperseg*0.5))
    # label = scipy.io.loadmat('./label.mat')['timit_label']
    # data, _ = librosa.load(wav_dir, config.fs)
    #
    # aa = utils.Truelabel2Trueframe(label, config.nperseg, int(config.nperseg*0.5))
    # pred = pred[0:aa.shape[0], :]
    # aa = np.mean(np.equal(aa, pred))
    # print(aa)
    #
    # aaa =1