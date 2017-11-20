"""
License: Apache-2.0
Author: Suofei Zhang
E-mail: zhangsuofei at njupt.edu.cn
"""

import tensorflow as tf
import config as cfg
from utils import create_inputs
import time
import numpy as np
import os
import capsnet_em as net


def main(_):
    with tf.Graph().as_default():
        batch_x, batch_labels = create_inputs(is_train=False)
        output, _ = net.build_arch(batch_x, is_train=False)
        batch_acc = net.test_accuracy(output, batch_labels)
        saver = tf.train.Saver()

        num_batches_per_epoch_train = int(60000 / cfg.batch_size)
        num_batches_test = int(10000/128)
        step = 0

        summaries = []
        summaries.append(tf.summary.scalar('accuracy', batch_acc))
        summary_op = tf.summary.merge(summaries)

        with tf.Session as sess:
            tf.train.start_queue_runners(sess=sess)
            summary_writer = tf.summary.FileWriter(cfg.test_logdir, graph=sess.graph)

            for epoch in range(cfg.epoch):
                ckpt = os.path.join(cfg.logdir, 'model.ckpt-%d' % (num_batches_per_epoch_train*epoch))
                saver.restore(sess, ckpt)

                for i in range(num_batches_test):
                    summary_str = sess.run(summary_op)
                    summary_writer.add_summary(summary_str, step)

                    step += 1

if __name__ == "__main__":
    tf.app.run()
