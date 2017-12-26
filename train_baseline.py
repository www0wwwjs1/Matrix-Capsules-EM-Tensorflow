"""
License: Apache-2.0
Author: Suofei Zhang | Hang Yu
E-mail: zhangsuofei at njupt.edu.cn | hangyu5 at illinois.edu
"""

import tensorflow as tf
import tensorflow.contrib.slim as slim
from config import cfg, get_coord_add, get_dataset_size_train, get_num_classes, get_create_inputs
import time
import numpy as np
import sys
import os
import capsnet_em as net

import logging
import daiquiri

daiquiri.setup(level=logging.DEBUG)
logger = daiquiri.getLogger(__name__)


def main(args):
    """Get dataset hyperparameters."""
    assert len(args) == 2 and isinstance(args[1], str)
    dataset_name = args[1]
    logger.info('Using dataset: {}'.format(dataset_name))

    """Set reproduciable random seed"""
    tf.set_random_seed(1234)

    coord_add = get_coord_add(dataset_name)
    dataset_size = get_dataset_size_train(dataset_name)
    num_classes = get_num_classes(dataset_name)
    create_inputs = get_create_inputs(dataset_name, is_train=True, epochs=cfg.epoch)

    with tf.Graph().as_default(), tf.device('/cpu:0'):
        """Get global_step."""
        global_step = tf.get_variable(
            'global_step', [], initializer=tf.constant_initializer(0), trainable=False)

        """Get batches per epoch."""
        num_batches_per_epoch = int(dataset_size / cfg.batch_size)

        """Use exponential decay leanring rate?"""
        lrn_rate = tf.maximum(tf.train.exponential_decay(
            1e-3, global_step, num_batches_per_epoch, 0.8), 1e-5)
        tf.summary.scalar('learning_rate', lrn_rate)
        opt = tf.train.AdamOptimizer()  # lrn_rate

        """Get batch from data queue."""
        batch_x, batch_labels = create_inputs()
        # batch_y = tf.one_hot(batch_labels, depth=10, axis=1, dtype=tf.float32)

        """Define the dataflow graph."""
        with tf.device('/gpu:0'):
            with slim.arg_scope([slim.variable], device='/cpu:0'):
                batch_x_squash = tf.divide(batch_x, 255.)
                batch_x = slim.batch_norm(batch_x, center=False, is_training=True, trainable=True)
                output = net.build_arch_baseline(batch_x, is_train=True,
                                                 num_classes=num_classes)
                loss, recon_loss, _ = net.cross_ent_loss(output, batch_x_squash, batch_labels)
                acc = net.test_accuracy(output, batch_labels)
                tf.summary.scalar('train_acc', acc)
                tf.summary.scalar('recon_loss', recon_loss)
                tf.summary.scalar('all_loss', loss)

            """Compute gradient."""
            grad = opt.compute_gradients(loss)
            # See: https://stackoverflow.com/questions/40701712/how-to-check-nan-in-gradients-in-tensorflow-when-updating
            grad_check = [tf.check_numerics(g, message='Gradient NaN Found!')
                          for g, _ in grad if g is not None] + [tf.check_numerics(loss, message='Loss NaN Found')]

        """Apply graident."""
        with tf.control_dependencies(grad_check):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = opt.apply_gradients(grad, global_step=global_step)

        """Set Session settings."""
        sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True, log_device_placement=False))
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())

        """Set Saver."""
        var_to_save = [v for v in tf.global_variables(
        ) if 'Adam' not in v.name]  # Don't save redundant Adam beta/gamma
        saver = tf.train.Saver(var_list=var_to_save, max_to_keep=cfg.epoch)

        """Display parameters"""
        total_p = np.sum([np.prod(v.get_shape().as_list()) for v in var_to_save]).astype(np.int32)
        train_p = np.sum([np.prod(v.get_shape().as_list())
                          for v in tf.trainable_variables()]).astype(np.int32)
        logger.info('Total Parameters: {}'.format(total_p))
        logger.info('Trainable Parameters: {}'.format(train_p))

        # read snapshot
        # latest = os.path.join(cfg.logdir, 'model.ckpt-4680')
        # saver.restore(sess, latest)
        """Set summary op."""
        summary_op = tf.summary.merge_all()

        """Start coord & queue."""
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        """Set summary writer"""
        if not os.path.exists(cfg.logdir + '/cnn_baseline/{}/train_log/'.format(dataset_name)):
            os.makedirs(cfg.logdir + '/cnn_baseline/{}/train_log/'.format(dataset_name))
        summary_writer = tf.summary.FileWriter(
            cfg.logdir + '/cnn_baseline/{}/train_log/'.format(dataset_name), graph=sess.graph)

        """Main loop."""
        for step in range(cfg.epoch * num_batches_per_epoch + 1):
            tic = time.time()
            """"TF queue would pop batch until no file"""
            try:
                _, loss_value, summary_str = sess.run(
                    [train_op, loss, summary_op])
                logger.info('%d iteration finishs in ' % step + '%f second' %
                            (time.time() - tic) + ' loss=%f' % loss_value)
            except KeyboardInterrupt:
                sess.close()
                sys.exit()
            except tf.errors.InvalidArgumentError:
                logger.warning('%d iteration contains NaN gradients. Discard.' % step)
                continue
            else:
                """Write to summary."""
                if step % 5 == 0:
                    summary_writer.add_summary(summary_str, step)

                """Epoch wise linear annealling."""
                if (step % num_batches_per_epoch) == 0:

                    """Save model periodically"""
                    ckpt_path = os.path.join(
                        cfg.logdir + '/cnn_baseline/{}'.format(dataset_name), 'model-{:.4f}.ckpt'.format(loss_value))
                    saver.save(sess, ckpt_path, global_step=step)

        """Join threads"""
        coord.join(threads)


if __name__ == "__main__":
    tf.app.run()
