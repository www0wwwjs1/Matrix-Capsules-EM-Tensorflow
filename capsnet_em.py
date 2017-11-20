"""
License: Apache-2.0
Author: Suofei Zhang
E-mail: zhangsuofei at njupt.edu.cn
"""

import tensorflow as tf
import tensorflow.contrib.slim as slim
from config import cfg
import numpy as np

def cross_ent_loss(output, y):
    loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=output)
    regularization = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    loss = tf.add_n([loss]+regularization)

    return loss

# input should be a tensor with size as [batch_size, height, width, channels]
def kernel_tile(input, kernel, stride):
    input_shape = input.get_shape()

    tile_filter = np.zeros(shape=[kernel, kernel, input_shape[3], kernel*kernel], dtype=np.float32)
    for i in range(kernel):
        for j in range(kernel):
            tile_filter[i, j, :, i * kernel + j] = 1.0

    tile_filter_op = tf.constant(tile_filter, dtype=tf.float32)
    output = tf.nn.depthwise_conv2d(input, tile_filter_op, strides=[1, stride, stride, 1], padding='VALID')
    output_shape = output.get_shape()
    output = tf.reshape(output, shape=[int(output_shape[0]), int(output_shape[1]), int(output_shape[2]), int(input_shape[3]), kernel*kernel])
    output = tf.transpose(output, perm=[0, 1, 2, 4, 3])

    return output

# input should be a tensor with size as [batch_size, caps_num_i, 16]
def mat_transform(input, caps_num_c, regularizer, tag = False):
    batch_size = int(input.get_shape()[0])
    caps_num_i = int(input.get_shape()[1])
    output = tf.reshape(input, shape=[batch_size, caps_num_i, 1, 4, 4])
    # the output of capsule is miu, the mean of a Gaussian, and activation, the sum of probabilities
    # it has no relationship with the absolute values of w and votes
    # using weights with bigger stddev helps numerical stability
    w = slim.variable('w', shape=[1, caps_num_i, caps_num_c, 4, 4], dtype=tf.float32,
                      initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1.0),
                      regularizer=regularizer)

    w = tf.tile(w, [batch_size, 1, 1, 1, 1])
    output = tf.tile(output, [1, 1, caps_num_c, 1, 1])
    votes = tf.reshape(tf.matmul(output, w), [batch_size, caps_num_i, caps_num_c, 16])

    return votes

def build_arch(input, is_train=False):
    # xavier initialization is necessary here to provide higher stability
    # initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
    # instead of initializing bias with constant 0, a truncated normal initializer is exploited here for higher stability
    bias_initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01) #tf.constant_initializer(0.0)
    # The paper didnot mention any regularization, a common l2 regularizer to weights is added here
    weights_regularizer = tf.contrib.layers.l2_regularizer(5e-04)

    with slim.arg_scope([slim.conv2d], trainable=is_train, biases_initializer=bias_initializer, weights_regularizer=weights_regularizer):#weights_initializer=initializer,
        with tf.variable_scope('relu_conv1') as scope:
            output = slim.conv2d(input, num_outputs=cfg.A, kernel_size=[5, 5], stride=2, padding='VALID', scope=scope)

            assert output.get_shape() == [cfg.batch_size, 12, 12, 32]

        with tf.variable_scope('primary_caps') as scope:
            pose = slim.conv2d(output, num_outputs=cfg.B*16, kernel_size=[1, 1], stride=1, padding='VALID', scope=scope, activation_fn=None)
            activation = slim.conv2d(output, num_outputs=cfg.B, kernel_size=[1, 1], stride=1, padding='VALID', activation_fn=tf.nn.sigmoid)
            pose = tf.reshape(pose, shape=[cfg.batch_size, 12, 12, cfg.B, 16])
            activation = tf.reshape(activation, shape=[cfg.batch_size, 12, 12, cfg.B, 1])
            output = tf.concat([pose, activation], axis=4)
            output = tf.reshape(output, shape=[cfg.batch_size, 12, 12, -1])
            assert output.get_shape() == [cfg.batch_size, 12, 12, cfg.B*17]

        with tf.variable_scope('conv_caps1') as scope:
            output = kernel_tile(output, 3, 2)
            output = tf.reshape(output, shape=[cfg.batch_size * 5 * 5, 3 * 3 * cfg.B, 17])
            activation = tf.reshape(output[:, :, 16], shape=[cfg.batch_size * 5 * 5, 3 * 3 * cfg.B, 1])

            with tf.variable_scope('v') as scope:
                votes = mat_transform(output[:, :, :16], cfg.C, weights_regularizer, tag=True)

            with tf.variable_scope('routing') as scope:
                miu, activation, _ = em_routing(votes, activation, cfg.C, weights_regularizer)

            pose = tf.reshape(miu, shape=[cfg.batch_size, 5, 5, cfg.C, 16])
            activation = tf.reshape(activation, shape=[cfg.batch_size, 5, 5, cfg.C, 1])
            output = tf.reshape(tf.concat([pose, activation], axis=4), [cfg.batch_size, 5, 5, -1])

        with tf.variable_scope('conv_caps2') as scope:
            output = kernel_tile(output, 3, 1)
            output = tf.reshape(output, shape=[cfg.batch_size * 3 * 3, 3 * 3 * cfg.C, 17])
            activation = tf.reshape(output[:, :, 16], shape=[cfg.batch_size * 3 * 3, 3 * 3 * cfg.C, 1])

            with tf.variable_scope('v') as scope:
                votes = mat_transform(output[:, :, :16], cfg.D, weights_regularizer)

            with tf.variable_scope('routing') as scope:
                miu, activation, _ = em_routing(votes, activation, cfg.D, weights_regularizer)

            pose = tf.reshape(miu, shape=[cfg.batch_size*3*3, cfg.D, 16])
            activation = tf.reshape(activation, shape=[cfg.batch_size*3*3, cfg.D, 1])

        # It is not clear from the paper that ConvCaps2 is full connected to Class Capsules, or is conv connected with kernel size of 1*1 and a global average pooling.
        # From the description in Figure 1 of the paper and the amount of parameters (310k in the paper and 316,853 in fact), I assume a conv cap plus a golbael average pooling is the design.
        # However, with this design, the introduction of scaled coordinate by Coordinate Addition technique is really puzzling.
        # How can a kernel exploits irrelevant coordinate information? So I annotate this part out temporarily.
        with tf.variable_scope('class_caps') as scope:
            # coords = np.zeros(shape=[3*3, 2], dtype=np.float32)
            # for i in range(3):
            #     for j in range(3):
            #         coords[i*3+j, 0] = ??? require calculation of the center of receptive field here
            #         coords[i*3+j, 1] = ???

            with tf.variable_scope('v') as scope:
                votes = mat_transform(pose, 10, weights_regularizer)
                # coords_op = tf.constant(coords, dtype=tf.float32)
                # coords_op = tf.reshape(coords_op, shape=[9, 1, 1, 2])
                # coords_op = tf.tile()
                assert votes.get_shape() == [cfg.batch_size * 3 * 3, cfg.D, 10, 16]

            with tf.variable_scope('routing') as scope:
                miu, activation, test1 = em_routing(votes, activation, 10, weights_regularizer, tag=True)

            output = tf.reshape(activation, shape=[cfg.batch_size, 3, 3, 10])

        output = tf.reshape(tf.nn.avg_pool(output, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='VALID'), shape=[cfg.batch_size, 10])

    return output, test1

def em_routing(votes, activation, caps_num_c, regularizer, tag=False):
    test = []

    batch_size = int(votes.get_shape()[0])
    caps_num_i = int(activation.get_shape()[1])

    # m-step
    r = tf.constant(np.ones([batch_size, caps_num_i, caps_num_c], dtype=np.float32) / 32)
    a = tf.tile(activation, [1, 1, caps_num_c])
    r = r * a

    r_sum = tf.reshape(tf.reduce_sum(r, axis=1), shape=[batch_size, 1, caps_num_c])
    r1 = tf.reshape(r / tf.tile(r_sum, [1, caps_num_i, 1]),
                    shape=[batch_size, caps_num_i, caps_num_c, 1])
    r1 = tf.tile(r1, [1, 1, 1, 16])
    r1 = tf.reshape(tf.transpose(r1, perm=[0, 2, 3, 1]),
                    shape=[batch_size, caps_num_c, 16, 1, caps_num_i])
    votes1 = tf.reshape(tf.transpose(votes, perm=[0, 2, 3, 1]),
                        shape=[batch_size, caps_num_c, 16, caps_num_i, 1])
    miu = tf.matmul(r1, votes1)

    miu_tile = tf.tile(miu, [1, 1, 1, caps_num_i, 1])
    sigma_square = tf.matmul(r1, tf.square(votes1 - miu_tile))+cfg.epsilon
    sigma_square = tf.reshape(sigma_square, [batch_size, caps_num_c, 16])

    beta_v = slim.variable('beta_v', shape=[caps_num_c, 16], dtype=tf.float32,
                           initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
                           regularizer=regularizer)
    beta_v_tile = tf.tile(tf.reshape(beta_v, shape=[1, caps_num_c, 16]), [batch_size, 1, 1])
    r_sum_tile = tf.tile(tf.reshape(r_sum, shape=[batch_size, caps_num_c, 1]), [1, 1, 16])
    cost_h = (beta_v_tile + tf.log(tf.sqrt(sigma_square))) * r_sum_tile

    beta_a = slim.variable('beta_a', shape=[caps_num_c], dtype=tf.float32,
                           initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
                           regularizer=regularizer)
    beta_a_tile = tf.tile(tf.reshape(beta_a, [1, caps_num_c]), [batch_size, 1])
    activation1 = tf.nn.sigmoid(cfg.ac_lambda0 * (beta_a_tile - tf.reduce_sum(cost_h, axis=2)))

    for iters in range(cfg.iter_routing):
        # e-step
        miu_tile = tf.tile(tf.reshape(miu, shape=[batch_size, 1, caps_num_c, 16]), [1, caps_num_i, 1, 1])
        sigma_square_tile = tf.tile(tf.reshape(sigma_square, shape=[batch_size, 1, caps_num_c, 16]),
                                    [1, caps_num_i, 1, 1])
        # algorithm from paper is replaced by products of p_{ch}, which supports better numerical stability
        p_c = 1/(tf.sqrt(2*3.14159*(sigma_square_tile)))*tf.exp(-tf.square(votes-miu_tile)/(2*(sigma_square_tile)))
        p_c = tf.reduce_prod(p_c, axis=3)

        # e_exp = tf.square(votes - miu_tile) / (2 * sigma_square_tile)
        # e_exp = -tf.reduce_sum(e_exp, axis=3)
        #
        # tmp = tf.reduce_prod(2 * 3.14159 * sigma_square, axis=2)
        # p_c = 1 / tf.sqrt(tmp)
        # p_c = tf.tile(tf.reshape(p_c, shape=[batch_size, 1, caps_num_c]), [1, caps_num_i, 1]) * e_exp

        a1_tile = tf.tile(tf.reshape(activation1, shape=[batch_size, 1, caps_num_c]), [1, caps_num_i, 1])
        ap = a1_tile * p_c
        sum_ap_tile = tf.tile(
            tf.reshape(tf.reduce_sum(ap, axis=2), shape=[batch_size, caps_num_i, 1]), [1, 1, caps_num_c])
        r = ap / sum_ap_tile

        # m-step
        r = r * a

        r_sum = tf.reshape(tf.reduce_sum(r, axis=1), shape=[batch_size, 1, caps_num_c])
        r1 = tf.reshape(r / tf.tile(r_sum, [1, caps_num_i, 1]),
                        shape=[batch_size, caps_num_i, caps_num_c, 1])
        r1 = tf.tile(r1, [1, 1, 1, 16])
        r1 = tf.reshape(tf.transpose(r1, perm=[0, 2, 3, 1]),
                        shape=[batch_size, caps_num_c, 16, 1, caps_num_i])

        miu = tf.matmul(r1, votes1)

        miu_tile = tf.tile(miu, [1, 1, 1, caps_num_i, 1])

        sigma_square = tf.matmul(r1, tf.square(votes1 - miu_tile))

        sigma_square = tf.reshape(sigma_square, [batch_size, caps_num_c, 16])

        r_sum_tile = tf.tile(tf.reshape(r_sum, shape=[batch_size, caps_num_c, 1]), [1, 1, 16])
        cost_h = (beta_v_tile + tf.log(tf.sqrt(sigma_square))) * r_sum_tile
        if tag:
            test.append(cost_h)

        activation1 = tf.nn.sigmoid(
            (cfg.ac_lambda0 + (iters + 1) * cfg.ac_lambda_step) * (beta_a_tile - tf.reduce_sum(cost_h, axis=2)))
        if tag:
            test.append(activation1)

    return miu, activation1, test

































