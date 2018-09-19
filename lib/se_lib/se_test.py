import matlab.engine

import sys
import subprocess

sys.path.insert(0, './lib')
import os
import lib.se_lib.datareader as dr
import lib.se_lib.graph_test as gt
import glob
import lib.se_lib.config as config
import scipy.io
import tensorflow as tf
import numpy as np
import lib.se_lib.utils as utils
import librosa
import librosa.display as ld
from time import sleep


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

clean_dir = '/home/jtkim/github/SE_data_test/data/test/clean'
noisy_dir = '/home/jtkim/SE_research/SE_pr_test'
norm_dir = './lib/se_lib/norm'
recon_save_dir = './test_result_pr'

# clean_dir = os.path.abspath('./data/test/clean')
# noisy_dir = os.path.abspath('./data/test/noisy')
# norm_dir = os.path.abspath('./data/train/norm')


class SE(object):

    def __init__(self, graph_name, target_fs, norm_path=norm_dir, save_dir = os.path.abspath('./enhanced_wav')):

        graph = gt.load_graph(graph_name)

        self.node_inputs = graph.get_tensor_by_name('prefix/model_1/inputs:0')
        self.node_labels = graph.get_tensor_by_name('prefix/model_1/labels:0')
        if config.mode != 'lstm' and config.mode != 'fcn':
            self.node_keep_prob = graph.get_tensor_by_name('prefix/model_1/keep_prob:0')
        self.node_prediction = graph.get_tensor_by_name('prefix/model_1/pred:0')

        sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess_config.gpu_options.allow_growth = True

        self.sess = tf.Session(config=sess_config, graph=graph)
        self.norm_path = norm_path
        self.save_dir = save_dir
        self.target_fs = target_fs

    def enhance(self, noisy_speech):

        noisy_speech = librosa.resample(noisy_speech, self.target_fs, 8000)
        # noisy_speech = utils.read_raw(wav_dir)
        temp_dir = './temp.npy'
        np.save(temp_dir, noisy_speech)

        test_dr = dr.DataReader(temp_dir, '', self.norm_path, dist_num=config.dist_num, is_training=False, is_shuffle=False)
        mean, std = test_dr.norm_process(self.norm_path + '/norm_noisy.mat')

        while True:
            test_inputs, test_labels, test_inphase, test_outphase = test_dr.whole_batch(test_dr.num_samples)
            if config.mode != 'lstm' and config.mode != 'fcn':
                feed_dict = {self.node_inputs: test_inputs, self.node_labels: test_labels, self.node_keep_prob: 1.0}
            else:
                feed_dict = {self.node_inputs: test_inputs, self.node_labels: test_labels}

            pred = self.sess.run(self.node_prediction, feed_dict=feed_dict)

            if test_dr.file_change_checker():

                lpsd = np.expand_dims(np.reshape(pred, [-1, config.freq_size]), axis=2)

                lpsd = np.squeeze((lpsd * std * config.global_std) + mean)

                recon_speech = utils.get_recon(np.transpose(lpsd, (1, 0)), np.transpose(test_inphase, (1, 0)),
                                               win_size=config.win_size, win_step=config.win_step, fs=config.fs)

                test_dr.reader_initialize()

                break

        # file_dir = self.save_dir + '/' + os.path.basename(wav_dir).replace('noisy', 'enhanced').replace('raw', 'wav')
        # librosa.output.write_wav(file_dir, recon_speech, int(config.fs), norm=True)

        recon_speech = librosa.resample(recon_speech, 8000, self.target_fs)
        os.popen('rm -rf temp.npy')

        return recon_speech


if __name__ == '__main__':

    # clean_dir = os.path.abspath('./data/test/clean')
    # noisy_dir = os.path.abspath('./data/test/noisy')
    # norm_dir = os.path.abspath('./data/train/norm')

    # logs_dir = os.path.abspath('./logs' + '/logs_' + "2018-06-04-02-06-49")
    model_dir = os.path.abspath('./saved_model')
    # gs.freeze_graph(logs_dir, model_dir, 'model_1/pred,model_1/labels,model_1/cost')

    graph_name = sorted(glob.glob(model_dir + '/*.pb'))[-1]
    # graph_name = '/home/jtkim/hdd3/github_2/SE_graph/Boost/Boost.pb'

    noisy_list = sorted(glob.glob(noisy_dir + '/*.raw'))
    num_data = len(noisy_list)

    se = SE(graph_name=graph_name, norm_path=norm_dir)

    computation_time = []
    for noisy_dir in noisy_list:
        fname = recon_save_dir + '/' + os.path.basename(noisy_dir).replace('.raw', '.wav')
        print(noisy_dir)

        # recon_speech = speech_enhance(noisy_dir, graph_name)
        start_time = time.time()
        recon_speech = se.enhance(noisy_dir)
        computation_time.append((time.time() - start_time) / (recon_speech.shape[0]/config.fs) * 1000)
        librosa.output.write_wav(fname, recon_speech, int(config.fs), norm=True)

    print(np.mean(np.asarray(computation_time)))

        # noisy_speech = utils.identity_trans(utils.read_raw(noisy_dir))
