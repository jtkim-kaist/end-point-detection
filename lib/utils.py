import tensorflow as tf
import numpy as np


def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out


def leakyrelu(x, alpha=0.3, name='lrelu'):
    return tf.maximum(x, alpha * x, name=name)


def conv2d_basic(x, W, bias, stride=[1,1], padding="SAME"):
    # conv = tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding=padding)
    conv = tf.nn.conv2d(x, W, strides=[1, stride[0], stride[1],1], padding=padding)

    return tf.nn.bias_add(conv, bias)


def conv_with_bn(inputs, out_channels, filter_size=[3, 3], stride=[1,1], act='relu', is_training=True,
                 padding="SAME", name=None):

    in_height = filter_size[0]
    in_width = filter_size[1]
    in_channels = inputs.get_shape().as_list()[3]
    W = weight_variable([in_height, in_width, in_channels, out_channels], name=name+'_W', is_training=is_training)
    b = bias_variable([out_channels], name=name+'_b', is_training=is_training)
    conv = conv2d_basic(inputs, W, b, stride=stride, padding=padding)
    conv = tf.contrib.layers.batch_norm(conv, decay=0.9, is_training=is_training, updates_collections=None)
    if act is 'relu':
        relu = tf.nn.relu(conv)
        # relu = leakyrelu(conv)

    if act is 'sigmoid':
        relu = tf.nn.sigmoid(conv)
    return relu


def conv_no_bn(inputs, out_channels, filter_size=[3, 3], stride=[1,1], act='relu', is_training=True,
                 padding="SAME", name=None):

    in_height = filter_size[0]
    in_width = filter_size[1]
    in_channels = inputs.get_shape().as_list()[3]
    W = weight_variable([in_height, in_width, in_channels, out_channels], name=name+'_W',is_training=is_training)
    b = bias_variable([out_channels], name=name+'_b',is_training=is_training)
    conv = conv2d_basic(inputs, W, b, stride=stride, padding=padding)
    if act is 'relu':
        relu = tf.nn.relu(conv)
        # relu = leakyrelu(conv)
    if act is 'sigmoid':
        relu = tf.nn.sigmoid(conv)
    return relu


def affine_transform(x, output_dim, name=None):
    """
    affine transformation Wx+b
    assumes x.shape = (batch_size, num_features)
    """

    w = tf.get_variable(name + "_w", [x.get_shape()[1], output_dim],
                        initializer=tf.truncated_normal_initializer(stddev=0.02))
    b = tf.get_variable(name + "_b", [output_dim], initializer=tf.constant_initializer(0.0))

    return tf.matmul(x, w) + b


def weight_variable(shape, stddev=0.02, name=None, is_training=True):

    initial = tf.truncated_normal(shape, stddev=stddev)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.get_variable(name, initializer=initial, trainable = is_training)


def bias_variable(shape, name=None, is_training=True):
    initial = tf.constant(0.0, shape=shape)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.get_variable(name, initializer=initial, trainable = is_training)


def frame2rawlabel(label, win_len, win_step):

    num_frame = label.shape[0]

    total_len = (num_frame-1) * win_step + win_len
    raw_label = np.zeros((total_len, 1))
    start_indx = 0

    i = 0

    while True:

        if start_indx + win_len > total_len:
            break
        else:
            temp_label = label[i]
            raw_label[start_indx+1:start_indx+win_len] = raw_label[start_indx+1:start_indx+win_len] + temp_label
        i += 1

        start_indx = start_indx + win_step

    raw_label = (raw_label >= 1).choose(raw_label, 1)

    return raw_label


def Truelabel2Trueframe( TrueLabel_bin,wsize,wstep ):
    iidx = 0
    Frame_iidx = 0
    Frame_len = Frame_Length(TrueLabel_bin, wstep, wsize)
    Detect = np.zeros([Frame_len, 1])
    while 1 :
        if iidx+wsize <= len(TrueLabel_bin) :
            TrueLabel_frame = TrueLabel_bin[iidx:iidx + wsize - 1]*10
        else:
            TrueLabel_frame = TrueLabel_bin[iidx:]*10

        if (np.sum(TrueLabel_frame) >= wsize / 2) :
            TrueLabel_frame = 1
        else :
            TrueLabel_frame = 0

        if (Frame_iidx >= len(Detect)):
            break

        Detect[Frame_iidx] = TrueLabel_frame
        iidx = iidx + wstep
        Frame_iidx = Frame_iidx + 1
        if (iidx > len(TrueLabel_bin)):
            break

    return Detect


def Frame_Length( x,overlap,nwind ):
    nx = len(x)
    noverlap = nwind - overlap
    framelen = int((nx - noverlap) / (nwind - noverlap))
    return framelen