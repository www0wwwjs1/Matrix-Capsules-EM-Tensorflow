"""
License: Apache-2.0
Author: Suofei Zhang
E-mail: zhangsuofei at njupt.edu.cn
"""

import tensorflow as tf
from config import cfg
from utils import create_inputs
import time
import numpy as np
import os
import capsnet_em as net

def main(_):
    with tf.Graph().as_default():
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

        num_batches_per_epoch = int(60000/cfg.batch_size)
        opt = tf.train.AdamOptimizer()

        batch_x, batch_labels = create_inputs(is_train=True)
        # batch_y = tf.one_hot(batch_labels, depth=10, axis=1, dtype=tf.float32)

        output = net.build_arch(batch_x, is_train=True)

        loss = net.cross_ent_loss(output, batch_labels)
        loss_name = 'cross_ent_loss'

        summaries = []
        summaries.append(tf.summary.scalar(loss_name, loss))

        grad = opt.compute_gradients(loss)
        train_op = opt.apply_gradients(grad, global_step=global_step)

        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver(tf.global_variables(), max_to_keep=cfg.epoch)

        #read snapshot
        # latest = os.path.join(cfg.logdir, 'model.ckpt-4680')
        # saver.restore(sess, latest)

        summary_op = tf.summary.merge(summaries)
        tf.train.start_queue_runners(sess=sess)

        summary_writer = tf.summary.FileWriter(cfg.logdir, graph=sess.graph)

        for step in range(cfg.epoch*num_batches_per_epoch):
            tic = time.time()
            _, loss_value = sess.run([train_op, loss])
            print('%d iteration is finished in ' % step + '%f second' % (time.time()-tic))
            # test1_v = sess.run(test2)

            # if np.isnan(loss_value):
            #     print('bbb')
            #  assert not np.isnan(np.any(test2_v[0])), 'a is nan'
            assert not np.isnan(loss_value), 'loss is nan'

            if step % 10 == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)

            if (step % num_batches_per_epoch) == 0:
                ckpt_path = os.path.join(cfg.logdir, 'model.ckpt')
                saver.save(sess, ckpt_path, global_step=step)

if __name__ == "__main__":
    tf.app.run()