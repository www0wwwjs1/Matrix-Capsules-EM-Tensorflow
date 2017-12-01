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
    coord_add = get_coord_add(dataset_name)
    dataset_size = get_dataset_size_train(dataset_name)
    num_classes = get_num_classes(dataset_name)
    create_inputs = get_create_inputs(dataset_name, is_train=True, epochs=cfg.epoch)

    """Set reproduciable random seed"""
    tf.set_random_seed(1234)

    with tf.Graph().as_default(), tf.device('/cpu:0'):
        """Get global_step."""
        global_step = tf.get_variable(
            'global_step', [], initializer=tf.constant_initializer(0), trainable=False)

        """Get batches per epoch."""
        num_batches_per_epoch = int(dataset_size / cfg.batch_size)

        """Set tf summaries."""
        summaries = []

        """Use exponential decay leanring rate?"""
        # lrn_rate = tf.maximum(tf.train.exponential_decay(1e-3, global_step, 2e2, 0.66), 1e-5)
        # summaries.append(tf.summary.scalar('learning_rate', lrn_rate))
        opt = tf.train.AdamOptimizer()#lrn_rate

        """Get batch from data queue."""
        batch_x, batch_labels = create_inputs()
        # batch_y = tf.one_hot(batch_labels, depth=10, axis=1, dtype=tf.float32)

        """Define the dataflow graph."""
        m_op = tf.placeholder(dtype=tf.float32, shape=())
        with tf.device('/gpu:0'):
            with slim.arg_scope([slim.variable], device='/cpu:0'):
                output = net.build_arch(batch_x, coord_add, is_train=True,
                                        num_classes=num_classes)
                # loss = net.cross_ent_loss(output, batch_labels)
                loss = net.spread_loss(output, batch_labels, m_op)

            """Compute gradient."""
            grad = opt.compute_gradients(loss)

        """Add loss to summary."""
        summaries.append(tf.summary.scalar('spread_loss', loss))

        """Apply graident."""
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
        summary_op = tf.summary.merge(summaries)

        """Start coord & queue."""
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        """Set summary writer"""
        summary_writer = tf.summary.FileWriter(cfg.logdir, graph=None)  # graph = sess.graph, huge!

        """Main loop."""
        m_min = 0.2
        m_max = 0.9
        m = m_min
        for step in range(cfg.epoch * num_batches_per_epoch):
            tic = time.time()
            """"TF queue would pop batch until no file"""
            _, loss_value = sess.run([train_op, loss], feed_dict={m_op: m})
            logger.info('%d iteration finishs in ' % step + '%f second' %
                        (time.time() - tic) + ' loss=%f' % loss_value)

            """Check NaN"""
            assert not np.isnan(loss_value), 'loss is nan'

            """Write to summary."""
            if step % 10 == 0:
                summary_str = sess.run(summary_op, feed_dict={m_op: m})
                summary_writer.add_summary(summary_str, step)

            """Epoch wise linear annealling."""
            if (step % num_batches_per_epoch) == 0:
                if step > 0:
                    m += (m_max - m_min) / (cfg.epoch * 0.6)
                    if m > m_max:
                        m = m_max

                """Save model periodically"""
                ckpt_path = os.path.join(cfg.logdir, 'model-{}.ckpt'.format(round(loss_value, 4)))
                saver.save(sess, ckpt_path, global_step=step)

        """Join threads"""
        coord.join(threads)


if __name__ == "__main__":
    tf.app.run()
