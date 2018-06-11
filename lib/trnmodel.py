import tensorflow as tf
from lib import utils
from lib import config
import matplotlib


class Model(object):

    def __init__(self, global_step, is_training=True,keep_prob = 1.,mode = 0):
        self.logit_name = 'CNN'
        self._num_layers = 2
        self.keep_prob=keep_prob
        self._is_training = is_training

        # self.keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
        self.inputs = inputs = tf.placeholder(tf.float32, shape=[None, config.time_width, config.n_mels, 1],
                                              name="inputs")
        self.labels = labels = tf.placeholder(tf.int64, shape=[None], name="labels")


        logits= self.inference(inputs)

        self.softpred = tf.identity(tf.nn.softmax(logits)[:, -1], name="softpred")
        self.pred = pred = tf.argmax(logits, axis=1, name="prediction")
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(pred, labels), tf.float32))
        self.Tcost = Tcost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
        # trainable_var = tf.trainable_variables()
        trainable_var = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope="model/CNN")

        self.train_op = self.train(Tcost, trainable_var, global_step)




    def inference(self, inputs):
        with tf.variable_scope(self.logit_name):
            i=0
            layer_name = 'T_conv_%d' % (0+i)
            fm= utils.conv_no_bn(inputs, out_channels=128, filter_size=[3, 3],
                                    stride=[1,1], act='relu', is_training=self._is_training,
                                    padding="SAME", name=layer_name)
            fm = tf.nn.max_pool(fm, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding='SAME')
            layer_name = 'T_conv_%d' % (1+i)
            fm2 = utils.conv_no_bn(fm, out_channels=128, filter_size=[3, 3],
                                  stride=[1,1], act='relu', is_training=self._is_training,
                                  padding="SAME", name=layer_name)
            fm2 = tf.nn.max_pool(fm2, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding='SAME')

            layer_name = 'T_conv_%d' % (2+i)
            fm3 = utils.conv_no_bn(fm2, out_channels=128, filter_size=[1, 1], stride=[1,1], act='relu',
                                    is_training=self._is_training, padding="SAME", name=layer_name)
            fm3 = tf.nn.max_pool(fm3, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding='SAME')


            layer_name = 'T_conv_%d' % (11+i)
            last_shape = fm3.get_shape().as_list()
            fm4= utils.conv_no_bn(fm3, out_channels=128, filter_size=[last_shape[1], last_shape[2]],
                                           stride=[1,1], act='relu', is_training=self._is_training, padding="VALID", name=layer_name)
            layer_name = 'T_conv_%d' % (12+i)
            fm5 = utils.conv_no_bn(fm4, out_channels=128, filter_size=[1, 1], stride=[1,1], act='relu',
                                    is_training=self._is_training, padding="SAME", name=layer_name)
            layer_name = 'T_conv_%d' % (13+i)
            fm6 = utils.conv_no_bn(fm5, out_channels=2, filter_size=[1, 1], stride=[1,1], act='relu',
                                    is_training=self._is_training, padding="SAME", name=layer_name)
            fm6 = tf.squeeze(fm6, [1, 2])
            logits = fm6

            return logits



    @staticmethod
    def train(loss, var_list, global_step):

        lrDecayRate = .96
        lrDecayFreq = 200
        momentumValue = .9

        # global_step = tf.Variable(0, trainable=False)
        lr = tf.train.exponential_decay(config.lr, global_step, lrDecayFreq, lrDecayRate, staircase=True)

        # define the optimizer
        # optimizer = tf.train.MomentumOptimizer(lr, momentumValue)
        # optimizer = tf.train.AdagradOptimizer(lr)
        #
        optimizer = tf.train.AdamOptimizer(lr)
        grads = optimizer.compute_gradients(loss, var_list=var_list)

        return optimizer.apply_gradients(grads, global_step=global_step)

if __name__ == '__main__':

    global_step = tf.Variable(0, trainable=False)

    Model(is_training=True, global_step=global_step)
