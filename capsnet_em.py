"""
License: Apache-2.0
Author: Suofei Zhang | Hang Yu
E-mail: zhangsuofei at njupt.edu.cn | hangyu5 at illinois.edu
"""

import tensorflow as tf
import tensorflow.contrib.slim as slim
from config import cfg
import numpy as np

def cross_ent_loss(output, x, y):
    loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=output)
    loss = tf.reduce_mean(loss)
    num_class = int(output.get_shape()[-1])
    data_size = int(x.get_shape()[1])

    # reconstruction loss
    y = tf.one_hot(y, num_class, dtype=tf.float32)
    y = tf.expand_dims(y, axis=2)
    output = tf.expand_dims(output, axis=2)
    output = tf.reshape(tf.multiply(output, y), shape=[cfg.batch_size, -1])
    tf.logging.info("decoder input value dimension:{}".format(output.get_shape()))

    with tf.variable_scope('decoder'):
        output = slim.fully_connected(output, 512, trainable=True)
        output = slim.fully_connected(output, 1024, trainable=True)
        output = slim.fully_connected(output, data_size * data_size,
                                      trainable=True, activation_fn=tf.sigmoid)

        x = tf.reshape(x, shape=[cfg.batch_size, -1])
        reconstruction_loss = tf.reduce_mean(tf.square(output - x))

    # regularization loss
    regularization = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    # loss+0.0005*reconstruction_loss+regularization#
    loss_all = tf.add_n([loss] + [0.0005 * reconstruction_loss] + regularization)

    return loss_all, reconstruction_loss, output


def spread_loss(output, pose_out, x, y, m):
    """
    # check NaN
    # See: https://stackoverflow.com/questions/40701712/how-to-check-nan-in-gradients-in-tensorflow-when-updating
    output_check = [tf.check_numerics(output, message='NaN Found!')]
    with tf.control_dependencies(output_check):
    """

    num_class = int(output.get_shape()[-1])
    data_size = int(x.get_shape()[1])

    y = tf.one_hot(y, num_class, dtype=tf.float32)

    # spread loss
    output1 = tf.reshape(output, shape=[cfg.batch_size, 1, num_class])
    y = tf.expand_dims(y, axis=2)
    at = tf.matmul(output1, y)
    """Paper eq(5)."""
    loss = tf.square(tf.maximum(0., m - (at - output1)))
    loss = tf.matmul(loss, 1. - y)
    loss = tf.reduce_mean(loss)

    # reconstruction loss
    # pose_out = tf.reshape(tf.matmul(pose_out, y, transpose_a=True), shape=[cfg.batch_size, -1])
    pose_out = tf.reshape(tf.multiply(pose_out, y), shape=[cfg.batch_size, -1])
    tf.logging.info("decoder input value dimension:{}".format(pose_out.get_shape()))

    with tf.variable_scope('decoder'):
        pose_out = slim.fully_connected(pose_out, 512, trainable=True, weights_regularizer=tf.contrib.layers.l2_regularizer(5e-04))
        pose_out = slim.fully_connected(pose_out, 1024, trainable=True, weights_regularizer=tf.contrib.layers.l2_regularizer(5e-04))
        pose_out = slim.fully_connected(pose_out, data_size * data_size,
                                        trainable=True, activation_fn=tf.sigmoid, weights_regularizer=tf.contrib.layers.l2_regularizer(5e-04))

        x = tf.reshape(x, shape=[cfg.batch_size, -1])
        reconstruction_loss = tf.reduce_mean(tf.square(pose_out - x))

    if cfg.weight_reg:
        # regularization loss
        regularization = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        # loss+0.0005*reconstruction_loss+regularization#
        loss_all = tf.add_n([loss] + [0.0005 * data_size* data_size * reconstruction_loss] + regularization)
    else:
        loss_all = tf.add_n([loss] + [0.0005 * data_size* data_size * reconstruction_loss])

    return loss_all, loss, reconstruction_loss, pose_out

# input should be a tensor with size as [batch_size, height, width, channels]


def kernel_tile(input, kernel, stride):
    # output = tf.extract_image_patches(input, ksizes=[1, kernel, kernel, 1], strides=[1, stride, stride, 1], rates=[1, 1, 1, 1], padding='VALID')

    input_shape = input.get_shape()
    tile_filter = np.zeros(shape=[kernel, kernel, input_shape[3],
                                  kernel * kernel], dtype=np.float32)
    for i in range(kernel):
        for j in range(kernel):
            tile_filter[i, j, :, i * kernel + j] = 1.0

    tile_filter_op = tf.constant(tile_filter, dtype=tf.float32)
    output = tf.nn.depthwise_conv2d(input, tile_filter_op, strides=[
                                    1, stride, stride, 1], padding='VALID')
    output_shape = output.get_shape()
    output = tf.reshape(output, shape=[int(output_shape[0]), int(
        output_shape[1]), int(output_shape[2]), int(input_shape[3]), kernel * kernel])
    output = tf.transpose(output, perm=[0, 1, 2, 4, 3])

    return output

# input should be a tensor with size as [batch_size, caps_num_i, 16]
def mat_transform(input, caps_num_c, regularizer, tag=False):
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


def build_arch_baseline(input, is_train: bool, num_classes: int):

    bias_initializer = tf.truncated_normal_initializer(
        mean=0.0, stddev=0.01)  # tf.constant_initializer(0.0)
    # The paper didnot mention any regularization, a common l2 regularizer to weights is added here
    weights_regularizer = tf.contrib.layers.l2_regularizer(5e-04)

    tf.logging.info('input shape: {}'.format(input.get_shape()))

    # weights_initializer=initializer,
    with slim.arg_scope([slim.conv2d, slim.fully_connected], trainable=is_train, biases_initializer=bias_initializer, weights_regularizer=weights_regularizer):
        with tf.variable_scope('relu_conv1') as scope:
            output = slim.conv2d(input, num_outputs=32, kernel_size=[
                                 5, 5], stride=1, padding='SAME', scope=scope, activation_fn=tf.nn.relu)
            output = slim.max_pool2d(output, [2, 2], scope='max_2d_layer1')

            tf.logging.info('output shape: {}'.format(output.get_shape()))

        with tf.variable_scope('relu_conv2') as scope:
            output = slim.conv2d(output, num_outputs=64, kernel_size=[
                                 5, 5], stride=1, padding='SAME', scope=scope, activation_fn=tf.nn.relu)
            output = slim.max_pool2d(output, [2, 2], scope='max_2d_layer2')

            tf.logging.info('output shape: {}'.format(output.get_shape()))

        output = slim.flatten(output)
        output = slim.fully_connected(output, 1024, scope='relu_fc3', activation_fn=tf.nn.relu)
        tf.logging.info('output shape: {}'.format(output.get_shape()))
        output = slim.dropout(output, 0.5, scope='dp')
        output = slim.fully_connected(output, num_classes, scope='final_layer', activation_fn=None)
        tf.logging.info('output shape: {}'.format(output.get_shape()))
        return output


def build_arch(input, coord_add, is_train: bool, num_classes: int):
    test1 = []
    data_size = int(input.get_shape()[1])
    # xavier initialization is necessary here to provide higher stability
    # initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
    # instead of initializing bias with constant 0, a truncated normal initializer is exploited here for higher stability
    bias_initializer = tf.truncated_normal_initializer(
        mean=0.0, stddev=0.01)  # tf.constant_initializer(0.0)
    # The paper didnot mention any regularization, a common l2 regularizer to weights is added here
    weights_regularizer = tf.contrib.layers.l2_regularizer(5e-04)

    tf.logging.info('input shape: {}'.format(input.get_shape()))

    # weights_initializer=initializer,
    with slim.arg_scope([slim.conv2d], trainable=is_train, biases_initializer=bias_initializer, weights_regularizer=weights_regularizer):
        with tf.variable_scope('relu_conv1') as scope:
            output = slim.conv2d(input, num_outputs=cfg.A, kernel_size=[
                                 5, 5], stride=2, padding='VALID', scope=scope, activation_fn=tf.nn.relu)
            data_size = int(np.floor((data_size - 4) / 2))

            assert output.get_shape() == [cfg.batch_size, data_size, data_size, cfg.A]
            tf.logging.info('conv1 output shape: {}'.format(output.get_shape()))

        with tf.variable_scope('primary_caps') as scope:
            pose = slim.conv2d(output, num_outputs=cfg.B * 16,
                               kernel_size=[1, 1], stride=1, padding='VALID', scope=scope, activation_fn=None)
            activation = slim.conv2d(output, num_outputs=cfg.B, kernel_size=[
                                     1, 1], stride=1, padding='VALID', scope='primary_caps/activation', activation_fn=tf.nn.sigmoid)
            pose = tf.reshape(pose, shape=[cfg.batch_size, data_size, data_size, cfg.B, 16])
            activation = tf.reshape(
                activation, shape=[cfg.batch_size, data_size, data_size, cfg.B, 1])
            output = tf.concat([pose, activation], axis=4)
            output = tf.reshape(output, shape=[cfg.batch_size, data_size, data_size, -1])
            assert output.get_shape() == [cfg.batch_size, data_size, data_size, cfg.B * 17]
            tf.logging.info('primary capsule output shape: {}'.format(output.get_shape()))

        with tf.variable_scope('conv_caps1') as scope:
            output = kernel_tile(output, 3, 2)
            data_size = int(np.floor((data_size - 2) / 2))
            output = tf.reshape(output, shape=[cfg.batch_size *
                                               data_size * data_size, 3 * 3 * cfg.B, 17])
            activation = tf.reshape(output[:, :, 16], shape=[
                                    cfg.batch_size * data_size * data_size, 3 * 3 * cfg.B, 1])

            with tf.variable_scope('v') as scope:
                votes = mat_transform(output[:, :, :16], cfg.C, weights_regularizer, tag=True)
                tf.logging.info('conv cap 1 votes shape: {}'.format(votes.get_shape()))

            with tf.variable_scope('routing') as scope:
                miu, activation, _ = em_routing(votes, activation, cfg.C, weights_regularizer)
                tf.logging.info('conv cap 1 miu shape: {}'.format(miu.get_shape()))
                tf.logging.info('conv cap 1 activation before reshape: {}'.format(
                    activation.get_shape()))

            pose = tf.reshape(miu, shape=[cfg.batch_size, data_size, data_size, cfg.C, 16])
            tf.logging.info('conv cap 1 pose shape: {}'.format(pose.get_shape()))
            activation = tf.reshape(
                activation, shape=[cfg.batch_size, data_size, data_size, cfg.C, 1])
            tf.logging.info('conv cap 1 activation after reshape: {}'.format(
                activation.get_shape()))
            output = tf.reshape(tf.concat([pose, activation], axis=4), [
                                cfg.batch_size, data_size, data_size, -1])
            tf.logging.info('conv cap 1 output shape: {}'.format(output.get_shape()))

        with tf.variable_scope('conv_caps2') as scope:
            output = kernel_tile(output, 3, 1)
            data_size = int(np.floor((data_size - 2) / 1))
            output = tf.reshape(output, shape=[cfg.batch_size *
                                               data_size * data_size, 3 * 3 * cfg.C, 17])
            activation = tf.reshape(output[:, :, 16], shape=[
                                    cfg.batch_size * data_size * data_size, 3 * 3 * cfg.C, 1])

            with tf.variable_scope('v') as scope:
                votes = mat_transform(output[:, :, :16], cfg.D, weights_regularizer)
                tf.logging.info('conv cap 2 votes shape: {}'.format(votes.get_shape()))

            with tf.variable_scope('routing') as scope:
                miu, activation, _ = em_routing(votes, activation, cfg.D, weights_regularizer)

            pose = tf.reshape(miu, shape=[cfg.batch_size * data_size * data_size, cfg.D, 16])
            tf.logging.info('conv cap 2 pose shape: {}'.format(votes.get_shape()))
            activation = tf.reshape(
                activation, shape=[cfg.batch_size * data_size * data_size, cfg.D, 1])
            tf.logging.info('conv cap 2 activation shape: {}'.format(activation.get_shape()))

        # It is not clear from the paper that ConvCaps2 is full connected to Class Capsules, or is conv connected with kernel size of 1*1 and a global average pooling.
        # From the description in Figure 1 of the paper and the amount of parameters (310k in the paper and 316,853 in fact), I assume a conv cap plus a golbal average pooling is the design.
        with tf.variable_scope('class_caps') as scope:
            with tf.variable_scope('v') as scope:
                votes = mat_transform(pose, num_classes, weights_regularizer)

                assert votes.get_shape() == [cfg.batch_size * data_size *
                                             data_size, cfg.D, num_classes, 16]
                tf.logging.info('class cap votes original shape: {}'.format(votes.get_shape()))

                coord_add = np.reshape(coord_add, newshape=[data_size * data_size, 1, 1, 2])
                coord_add = np.tile(coord_add, [cfg.batch_size, cfg.D, num_classes, 1])
                coord_add_op = tf.constant(coord_add, dtype=tf.float32)

                votes = tf.concat([coord_add_op, votes], axis=3)
                tf.logging.info('class cap votes coord add shape: {}'.format(votes.get_shape()))

            with tf.variable_scope('routing') as scope:
                miu, activation, test2 = em_routing(
                    votes, activation, num_classes, weights_regularizer)
                tf.logging.info(
                    'class cap activation shape: {}'.format(activation.get_shape()))
                tf.summary.histogram(name="class_cap_routing_hist",
                                     values=test2)

            output = tf.reshape(activation, shape=[
                                cfg.batch_size, data_size, data_size, num_classes])

        output = tf.reshape(tf.nn.avg_pool(output, ksize=[1, data_size, data_size, 1], strides=[
                            1, 1, 1, 1], padding='VALID'), shape=[cfg.batch_size, num_classes])
        tf.logging.info('class cap output shape: {}'.format(output.get_shape()))

        pose = tf.nn.avg_pool(tf.reshape(miu, shape=[cfg.batch_size, data_size, data_size, -1]), ksize=[
                              1, data_size, data_size, 1], strides=[1, 1, 1, 1], padding='VALID')
        pose_out = tf.reshape(pose, shape=[cfg.batch_size, num_classes, 18])

    return output, pose_out


def test_accuracy(logits, labels):
    logits_idx = tf.to_int32(tf.argmax(logits, axis=1))
    logits_idx = tf.reshape(logits_idx, shape=(cfg.batch_size,))
    correct_preds = tf.equal(tf.to_int32(labels), logits_idx)
    accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32)) / cfg.batch_size

    return accuracy


def em_routing(votes, activation, caps_num_c, regularizer, tag=False):
    test = []

    batch_size = int(votes.get_shape()[0])
    caps_num_i = int(activation.get_shape()[1])
    n_channels = int(votes.get_shape()[-1])

    sigma_square = []
    miu = []
    activation_out = []
    beta_v = slim.variable('beta_v', shape=[caps_num_c, n_channels], dtype=tf.float32,
                           initializer=tf.constant_initializer(0.0),#tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
                           regularizer=regularizer)
    beta_a = slim.variable('beta_a', shape=[caps_num_c], dtype=tf.float32,
                           initializer=tf.constant_initializer(0.0),#tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
                           regularizer=regularizer)

    # votes_in = tf.stop_gradient(votes, name='stop_gradient_votes')
    # activation_in = tf.stop_gradient(activation, name='stop_gradient_activation')
    votes_in = votes
    activation_in = activation

    for iters in range(cfg.iter_routing):
        # if iters == cfg.iter_routing-1:

        # e-step
        if iters == 0:
            r = tf.constant(np.ones([batch_size, caps_num_i, caps_num_c], dtype=np.float32) / caps_num_c)
        else:
            # Contributor: Yunzhi Shi
            # log and exp here provide higher numerical stability especially for bigger number of iterations
            log_p_c_h = -tf.log(tf.sqrt(sigma_square)) - \
                        (tf.square(votes_in - miu) / (2 * sigma_square))
            log_p_c_h = log_p_c_h - \
                        (tf.reduce_max(log_p_c_h, axis=[2, 3], keep_dims=True) - tf.log(10.0))
            p_c = tf.exp(tf.reduce_sum(log_p_c_h, axis=3))

            ap = p_c * tf.reshape(activation_out, shape=[batch_size, 1, caps_num_c])

            # ap = tf.reshape(activation_out, shape=[batch_size, 1, caps_num_c])

            r = ap / (tf.reduce_sum(ap, axis=2, keep_dims=True) + cfg.epsilon)

        # m-step
        r = r * activation_in
        r = r / (tf.reduce_sum(r, axis=2, keep_dims=True)+cfg.epsilon)

        r_sum = tf.reduce_sum(r, axis=1, keep_dims=True)
        r1 = tf.reshape(r / (r_sum + cfg.epsilon),
                        shape=[batch_size, caps_num_i, caps_num_c, 1])

        miu = tf.reduce_sum(votes_in * r1, axis=1, keep_dims=True)
        sigma_square = tf.reduce_sum(tf.square(votes_in - miu) * r1,
                                     axis=1, keep_dims=True) + cfg.epsilon

        if iters == cfg.iter_routing-1:
            r_sum = tf.reshape(r_sum, [batch_size, caps_num_c, 1])
            cost_h = (beta_v + tf.log(tf.sqrt(tf.reshape(sigma_square,
                                                         shape=[batch_size, caps_num_c, n_channels])))) * r_sum

            activation_out = tf.nn.softmax(cfg.ac_lambda0 * (beta_a - tf.reduce_sum(cost_h, axis=2)))
        else:
            activation_out = tf.nn.softmax(r_sum)
        # if iters <= cfg.iter_routing-1:
        #     activation_out = tf.stop_gradient(activation_out, name='stop_gradient_activation')

    return miu, activation_out, test
