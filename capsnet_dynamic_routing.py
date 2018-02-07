"""
License: Apache-2.0
Author: Suofei Zhang
E-mail: zhangsuofei at njupt.edu.cn
"""

import tensorflow as tf
import tensorflow.contrib.slim as slim
from config import cfg
import numpy as np

# input should be a tensor with size as [batch_size, caps_num_in, channel_num_in]
def vec_transform(input, caps_num_out, channel_num_out):
    batch_size = int(input.get_shape()[0])
    caps_num_in = int(input.get_shape()[1])
    channel_num_in = int(input.get_shape()[-1])

    w = slim.variable('w', shape=[1, caps_num_out, caps_num_in, channel_num_in, channel_num_out], dtype=tf.float32,
                      initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01)) #

    w = tf.tile(w, [batch_size, 1, 1, 1, 1])
    output = tf.reshape(input, shape=[batch_size, 1, caps_num_in, 1, channel_num_in])
    output = tf.tile(output, [1, caps_num_out, 1, 1, 1])

    output = tf.reshape(tf.matmul(output, w), [batch_size, caps_num_out, caps_num_in, channel_num_out])

    return output

# input should be a tensor with size as [batch_size, caps_num_out, channel_num]
def squash(input):
    norm_2 = tf.reduce_sum(tf.square(input), axis=-1, keep_dims=True)
    output = norm_2/(tf.sqrt(norm_2+cfg.epsilon)*(1+norm_2))*input
    # output = tf.sqrt(norm_2+cfg.epsilon)/(1+norm_2)*input
    return output

# input should be a tensor with size as [batch_size, caps_num_out, caps_num_in, channel_num]
def dynamic_routing(input):
    batch_size = int(input.get_shape()[0])
    caps_num_in = int(input.get_shape()[2])
    caps_num_out = int(input.get_shape()[1])

    input_stopped = tf.stop_gradient(input, name='stop_gradient')

    b = tf.constant(np.zeros([batch_size, caps_num_out, caps_num_in, 1], dtype=np.float32))

    for r_iter in range(cfg.iter_routing):
        c = tf.nn.softmax(b, dim=1)
        if r_iter == cfg.iter_routing-1:
            s = tf.matmul(input, c, transpose_a=True)
            v = squash(tf.squeeze(s))
        else:
            s = tf.matmul(input_stopped, c, transpose_a=True)
            v = squash(tf.squeeze(s))
            b += tf.reduce_sum(tf.reshape(v, shape=[batch_size, caps_num_out, 1, -1])*input_stopped, axis=-1, keep_dims=True)

    return v

def build_arch(input, is_train, num_classes):
    data_size = int(input.get_shape()[1])
    initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
    bias_initializer = tf.constant_initializer(0.0)
    # weights_regularizer = tf.contrib.layers.l2_regularizer(5e-04)

    with slim.arg_scope([slim.conv2d], trainable=is_train, biases_initializer=bias_initializer, weights_initializer=initializer):#, activation_fn=None, , , , weights_regularizer=weights_regularizer
        with tf.variable_scope('conv1') as scope:
            output = slim.conv2d(input, num_outputs=256, kernel_size=[9, 9], stride=1, padding='VALID', scope=scope)
            data_size = data_size-8
            assert output.get_shape() == [cfg.batch_size, data_size, data_size, 256]
            tf.logging.info('conv1 output shape: {}'.format(output.get_shape()))

        with tf.variable_scope('primary_caps_layer') as scope:
            output = slim.conv2d(output, num_outputs=32*8, kernel_size=[9, 9], stride=2, padding='VALID', scope=scope)#, activation_fn=None
            output = tf.reshape(output, [cfg.batch_size, -1, 8])
            output = squash(output)
            data_size = int(np.floor((data_size-8)/2))
            assert output.get_shape() == [cfg.batch_size, data_size*data_size*32, 8]
            tf.logging.info('primary capsule output shape: {}'.format(output.get_shape()))

        with tf.variable_scope('digit_caps_layer') as scope:
            with tf.variable_scope('u') as scope:
                u_hats = vec_transform(output, num_classes, 16)
                assert u_hats.get_shape() == [cfg.batch_size, num_classes, data_size*data_size*32, 16]
                tf.logging.info('digit_caps_layer u_hats shape: {}'.format(u_hats.get_shape()))

            with tf.variable_scope('routing') as scope:
                output = dynamic_routing(u_hats)
                assert output.get_shape() == [cfg.batch_size, num_classes, 16]
                tf.logging.info('the output capsule has shape: {}'.format(output.get_shape()))

        output_len = tf.norm(output, axis=-1)

    return output, output_len

def loss(output, output_len, x, y):
    num_class = int(output_len.get_shape()[-1])
    data_size = int(x.get_shape()[1])
    y = tf.one_hot(y, num_class, dtype=tf.float32)

    # margin loss
    max_l = tf.square(tf.maximum(0., cfg.m_plus-output_len))
    max_r = tf.square(tf.maximum(0., output_len-cfg.m_minus))

    l_c = y*max_l+cfg.lambda_val*(1.-y)*max_r
    margin_loss = tf.reduce_mean(tf.reduce_sum(l_c, axis=1))

    # reconstruction loss
    y = tf.expand_dims(y, axis=2)
    # output = tf.squeeze(tf.matmul(output, y, transpose_a=True))
    output = tf.reshape(output*y, shape=[cfg.batch_size, -1])
    # test = output

    with tf.variable_scope('decoder'):
        output = slim.fully_connected(output, 512, trainable=True)
        output = slim.fully_connected(output, 1024, trainable=True)
        output = slim.fully_connected(output, data_size * data_size,
                                        trainable=True, activation_fn=tf.sigmoid)

        x = tf.reshape(x, shape=[cfg.batch_size, -1])
        reconstruction_loss = tf.reduce_mean(tf.square(output-x))

    if cfg.weight_reg:
        # regularization loss
        regularization = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        # loss+0.0005*reconstruction_loss+regularization#
        loss_all = tf.add_n([margin_loss] + [0.0005 * data_size * data_size * reconstruction_loss] + regularization)
    else:
        loss_all = margin_loss+0.0005*data_size*data_size*reconstruction_loss

    return loss_all, margin_loss, reconstruction_loss, output

def test_accuracy(logits, labels):
    logits_idx = tf.to_int32(tf.argmax(logits, axis=1))
    logits_idx = tf.reshape(logits_idx, shape=(cfg.batch_size,))
    correct_preds = tf.equal(tf.to_int32(labels), logits_idx)
    accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32)) / cfg.batch_size

    return accuracy


