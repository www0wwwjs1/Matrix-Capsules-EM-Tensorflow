"""
License: Apache-2.0
Author: Suofei Zhang
E-mail: zhangsuofei at njupt.edu.cn
"""

import tensorflow as tf
import tensorflow.contrib.slim as slim
from config import cfg
from utils import load_mnist
import time
import numpy as np
import os
import capsnet_em as net

def create_inputs():
    tr_x, tr_y = load_mnist(cfg.dataset, cfg.is_train)
    data_queue = tf.train.slice_input_producer([tr_x, tr_y], capacity=64*8)
    x, y = tf.train.shuffle_batch(data_queue, num_threads=8, batch_size=cfg.batch_size, capacity=cfg.batch_size*64,
                                  min_after_dequeue=cfg.batch_size*32, allow_smaller_final_batch=False)

    return (x, y)

def main(_):
    with tf.Graph().as_default():
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

        num_batches_per_epoch = int(60000/cfg.batch_size)
        opt = tf.train.AdamOptimizer()

        batch_x, batch_labels = create_inputs()
        # batch_y = tf.one_hot(batch_labels, depth=10, axis=1, dtype=tf.float32)

        output, test2 = net.build_arch(batch_x, is_train=True)

        loss = net.cross_ent_loss(output, batch_labels)
        loss_name = 'cross_ent_loss'

        summaries = []
        summaries.append(tf.summary.scalar(loss_name, loss))

        grad = opt.compute_gradients(loss)
        train_op = opt.apply_gradients(grad, global_step=global_step)

        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver(tf.global_variables(), max_to_keep=cfg.epoch)
        summary_op = tf.summary.merge(summaries)
        tf.train.start_queue_runners(sess=sess)

        summary_writer = tf.summary.FileWriter(cfg.logdir, graph=sess.graph)

        for step in range(cfg.epoch*num_batches_per_epoch):
            tic = time.time()
            _, loss_value = sess.run([train_op, loss])
            print('%d iteration is finished in' % step + '%f second' % (time.time()-tic))
            # test1_v = sess.run(test2)

            assert not np.isnan(loss_value), 'loss is nan'

            if step % 10 == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)

            if step / num_batches_per_epoch == 0:
                ckpt_path = os.path.join(cfg.logdir, 'model.ckpt')
                saver.save(sess, ckpt_path, global_step=step)

if __name__ == "__main__":
    tf.app.run()