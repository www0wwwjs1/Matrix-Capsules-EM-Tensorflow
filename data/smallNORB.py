import argparse
import argh
import sys
import os
import tensorflow as tf
import numpy as np

import logging
import daiquiri

daiquiri.setup(level=logging.DEBUG)
logger = daiquiri.getLogger(__name__)

from numpy.random import RandomState

prng = RandomState(1234567890)

from matplotlib import pyplot as plt
import cv2


def plot_imgs(inputs, num, label):
    """Plot smallNORB images helper"""
    # fig = plt.figure()
    # plt.title('Show images')
    # r = np.floor(np.sqrt(len(inputs))).astype(int)
    # for i in range(r**2):
    #     size = inputs[i].shape[1]
    #     sample = inputs[i].flatten().reshape(size, size)
    #     a = fig.add_subplot(r, r, i + 1)
    #     a.imshow(sample, cmap='gray')
    # plt.show()
    inputs = (inputs).astype(np.uint8)
    for i in range(len(inputs)):
        size = inputs[i].shape[1]
        cv2.imwrite('%d' % num+'_%d' % i+label+'.jpg', inputs[i].flatten().reshape(size, size))
    return


def write_data_to_tfrecord(kind: str, chunkify=False):
    """Credit: https://github.com/shashanktyagi/DC-GAN-on-smallNORB-dataset/blob/master/src/model.py
       Original Version: shashanktyagi
    """

    """Plan A: write dataset into one big tfrecord"""

    """Plan B: write dataset into manageable chuncks"""
    CHUNK = 24300 * 2 / 10  # create 10 chunks

    from time import time
    start = time()
    """Read data"""
    if kind == "train":
        fid_images = open('./smallNORB/smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat', 'rb')
        fid_labels = open('./smallNORB/smallnorb-5x46789x9x18x6x2x96x96-training-cat.mat', 'rb')
    elif kind == "test":
        fid_images = open('./smallNORB/smallnorb-5x01235x9x18x6x2x96x96-testing-dat.mat', 'rb')
        fid_labels = open('./smallNORB/smallnorb-5x01235x9x18x6x2x96x96-testing-cat.mat', 'rb')
    else:
        logger.warning('Please choose either training or testing data to preprocess.')

    logger.debug('Read data ' + kind + ' finish.')

    """Preprocessing"""
    for i in range(6):
        a = fid_images.read(4)  # header

    total_num_images = 24300 * 2

    for j in range(total_num_images // CHUNK if chunkify else 1):

        num_images = CHUNK if chunkify else total_num_images  # 24300 * 2
        images = np.zeros((num_images, 96 * 96))
        for idx in range(num_images):
            temp = fid_images.read(96 * 96)
            images[idx, :] = np.fromstring(temp, 'uint8')
        for i in range(5):
            a = fid_labels.read(4)  # header
        labels = np.fromstring(fid_labels.read(num_images * np.dtype('int32').itemsize), 'int32')
        labels = np.repeat(labels, 2)

        logger.debug('Load data %d finish. Start filling chunk %d.' % (j, j))

        # make dataset permuatation reproduceable
        perm = prng.permutation(num_images)
        images = images[perm]
        labels = labels[perm]

        """display image"""
        '''
        if j == 0:
            plot_imgs(images[:10])
        '''

        """Write to tfrecord"""
        writer = tf.python_io.TFRecordWriter("./" + kind + "%d.tfrecords" % j)
        for i in range(num_images):
            if i % 100 == 0:
                logger.debug('Write ' + kind + ' images %d' % ((j + 1) * i))
            img = images[i, :].tobytes()
            lab = labels[i].astype(np.int64)
            example = tf.train.Example(features=tf.train.Features(feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[lab])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img]))
            }))
            writer.write(example.SerializeToString())  # 序列化为字符串
        writer.close()

    # Should take less than a minute
    logger.info('Done writing ' + kind + '. Total time: %f' % (time() - start))


def tfrecord():
    """Wrapper"""
    write_data_to_tfrecord(kind='train', chunkify=False)
    write_data_to_tfrecord(kind='test', chunkify=False)
    logger.info('Writing train & test to TFRecord done.')


def read_norb_tfrecord(filenames, epochs: int):
    """Credit: http: // ycszen.github.io / 2016 / 08 / 17 / TensorFlow高效读取数据/
       Original Version: Ycszen-物语
    """

    assert isinstance(filenames, list)

    # 根据文件名生成一个队列
    filename_queue = tf.train.string_input_producer(filenames, num_epochs=epochs)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)  # 返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                       })
    img = tf.decode_raw(features['img_raw'], tf.float64)
    #logger.debug('Raw->img shape: {}'.format(img.get_shape()))
    img = tf.reshape(img, [96, 96, 1])
    img = tf.cast(img, tf.float32)  # * (1. / 255) # left unnormalized
    label = tf.cast(features['label'], tf.int32)
    # label = tf.one_hot(label, 5, dtype=tf.int32)  # left dense label
    #logger.debug('Raw->img shape: {}, label shape: {}'.format(img.get_shape(), label.get_shape()))
    return img, label


def test(is_train=True):
    """Instruction on how to read data from tfrecord"""

    # 1. use regular expression to find all files we want
    import re
    if is_train:
        CHUNK_RE = re.compile(r"train\d+\.tfrecords")
    else:
        CHUNK_RE = re.compile(r"test\d+\.tfrecords")

    processed_dir = './data'
    # 2. parse them into a list of file name
    chunk_files = [os.path.join(processed_dir, fname)
                   for fname in os.listdir(processed_dir)
                   if CHUNK_RE.match(fname)]
    # 3. pass argument into read method
    image, label = read_norb_tfrecord(chunk_files, 2)

    image = tf.image.random_brightness(image, max_delta=32. / 255.)
    image = tf.image.random_contrast(image, lower=0.5, upper=1.5)

    image = tf.image.resize_images(image, [48, 48])

    """Batch Norm"""
    params_shape = [image.get_shape()[-1]]
    beta = tf.get_variable(
        'beta', params_shape, tf.float32,
        initializer=tf.constant_initializer(0.0, tf.float32))
    gamma = tf.get_variable(
        'gamma', params_shape, tf.float32,
        initializer=tf.constant_initializer(1.0, tf.float32))
    mean, variance = tf.nn.moments(image, [0, 1, 2])
    image = tf.nn.batch_normalization(image, mean, variance, beta, gamma, 0.001)

    image = tf.random_crop(image, [32, 32, 1])

    batch_size = 8
    x, y = tf.train.shuffle_batch([image, label], batch_size=batch_size, capacity=batch_size * 64,
                                  min_after_dequeue=batch_size * 32, allow_smaller_final_batch=False)
    logger.debug('x shape: {}, y shape: {}'.format(x.get_shape(), y.get_shape()))

    # 初始化所有的op
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        sess.run(init)
        # 启动队列
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for i in range(2):
            val, l = sess.run([x, y])
            # l = to_categorical(l, 12)
            print(val, l)
        coord.join()

    logger.debug('Test read tf record Succeed')


parser = argparse.ArgumentParser()
argh.add_commands(parser, [tfrecord, test])

if __name__ == "__main__":
    argh.dispatch(parser)
