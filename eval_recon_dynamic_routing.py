"""
License: Apache-2.0
Author: Suofei Zhang | Hang Yu
E-mail: zhangsuofei at njupt.edu.cn | hangyu5 at illinois.edu
"""

import tensorflow as tf
from config import cfg, get_coord_add, get_dataset_size_train, get_dataset_size_test, get_num_classes, get_create_inputs
import time
import os
import capsnet_dynamic_routing as net
import tensorflow.contrib.slim as slim
from data.smallNORB import plot_imgs

import logging
import daiquiri

daiquiri.setup(level=logging.DEBUG)
logger = daiquiri.getLogger(__name__)


def main(args):
    """Get dataset hyperparameters."""
    assert len(args) == 3 and isinstance(args[1], str) and isinstance(args[2], str)
    dataset_name = args[1]
    model_name = args[2]

    """Set reproduciable random seed"""
    tf.set_random_seed(1234)

    coord_add = get_coord_add(dataset_name)
    dataset_size_train = get_dataset_size_train(dataset_name)
    dataset_size_test = get_dataset_size_test(dataset_name)
    num_classes = get_num_classes(dataset_name)
    create_inputs = get_create_inputs(
        dataset_name, is_train=False, epochs=cfg.epoch)

    with tf.Graph().as_default():
        num_batches_per_epoch_train = int(dataset_size_train / cfg.batch_size)
        num_batches_test = 2  # int(dataset_size_test / cfg.batch_size * 0.1)

        batch_x, batch_labels = create_inputs()
        batch_squash = tf.divide(batch_x, 255.)
        # batch_x_norm = slim.batch_norm(batch_x, center=False, is_training=False, trainable=False)
        pose_out, output = net.build_arch(batch_squash, is_train=False, num_classes=num_classes)
        # output, pose_out = net.build_arch(batch_x_norm, coord_add, is_train=False, num_classes=num_classes)
        tf.logging.debug(pose_out.get_shape())

        batch_acc = net.test_accuracy(output, batch_labels)
        # m_op = tf.constant(0.9)
        loss, margin_loss, mse, recon_img_squash = net.loss(
            pose_out, output, batch_squash, batch_labels)
        tf.summary.scalar('spread_loss', margin_loss)
        tf.summary.scalar('reconstruction_loss', mse)
        tf.summary.scalar('all_loss', loss)
        data_size = int(batch_x.get_shape()[1])
        recon_img = tf.multiply(tf.reshape(recon_img_squash, shape=[
                                cfg.batch_size, data_size, data_size, 1]), 255.)
        orig_img = tf.reshape(batch_x, shape=[
            cfg.batch_size, data_size, data_size, 1])
        tf.summary.image('orig_image', orig_img)
        tf.summary.image('recon_image', recon_img)
        saver = tf.train.Saver()

        step = 0

        tf.summary.scalar('accuracy', batch_acc)
        summary_op = tf.summary.merge_all()

        with tf.Session(config=tf.ConfigProto(
                allow_soft_placement=True, log_device_placement=False)) as sess:
            sess.run(tf.local_variables_initializer())
            sess.run(tf.global_variables_initializer())

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            if not os.path.exists(cfg.test_logdir + '/{}/{}/'.format(model_name, dataset_name)):
                os.makedirs(cfg.test_logdir + '/{}/{}/'.format(model_name, dataset_name))
            summary_writer = tf.summary.FileWriter(
                cfg.test_logdir + '/{}/{}/'.format(model_name, dataset_name), graph=sess.graph)  # graph=sess.graph, huge!

            files = os.listdir(cfg.logdir + '/{}/{}/'.format(model_name, dataset_name))
            for epoch in range(5, 6):
                # requires a regex to adapt the loss value in the file name here
                ckpt_re = ".ckpt-%d" % (num_batches_per_epoch_train * epoch)
                for __file in files:
                    if __file.endswith(ckpt_re + ".index"):
                        ckpt = os.path.join(
                            cfg.logdir + '/{}/{}/'.format(model_name, dataset_name), __file[:-6])
                # ckpt = os.path.join(cfg.logdir, "model.ckpt-%d" % (num_batches_per_epoch_train * epoch))
                saver.restore(sess, ckpt)

                accuracy_sum = 0
                for i in range(num_batches_test):
                    batch_acc_v, summary_str, orig_image, recon_image = sess.run(
                        [batch_acc, summary_op, orig_img, recon_img])
                    print('%d batches are tested.' % step)
                    summary_writer.add_summary(summary_str, step)

                    accuracy_sum += batch_acc_v

                    step += 1
                    # display original/reconstructed images in matplotlib
                    plot_imgs(orig_image, i, 'ori')
                    plot_imgs(recon_image, i, 'rec')

                ave_acc = accuracy_sum / num_batches_test
                print('the average accuracy is %f' % ave_acc)


if __name__ == "__main__":
    tf.app.run()
