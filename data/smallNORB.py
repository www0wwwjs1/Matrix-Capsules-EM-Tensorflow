import argparse
import argh
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


def plot_imgs(inputs):
    """Plot smallNORB images helper"""
    fig = plt.figure()
    plt.title('Show images')
    r = np.floor(np.sqrt(len(inputs))).astype(int)
    for i in range(r**2):
        sample = inputs[i, :].reshape(96, 96)
        a = fig.add_subplot(r, r, i + 1)
        a.imshow(sample)
    plt.show()


def write_data_to_tfrecord(kind: str, chunkify=False):
    """Credit: https://github.com/shashanktyagi/DC-GAN-on-NORB-dataset/blob/master/src/model.py
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

    logger.debug(f'Read data {kind} finish.')

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

        logger.debug(f'Load data {j} finish. Start filling chunk {j}.')

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
        writer = tf.python_io.TFRecordWriter(f"./{kind}{j}.tfrecords")
        for i in range(num_images):
            if i % 100 == 0:
                logger.debug(f'Write {kind} images {(j+1)*i}')
            img = images[i, :].tobytes()
            lab = labels[i].astype(np.int64)
            example = tf.train.Example(features=tf.train.Features(feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[lab])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img]))
            }))
            writer.write(example.SerializeToString())  # 序列化为字符串
        writer.close()

    # Should take less than a minute
    logger.info(f'Done writing {kind}. Total time: {time()-start}')


def tfrecord():
    """Wrapper"""
    write_data_to_tfrecord(kind='train', chunkify=False)
    write_data_to_tfrecord(kind='test', chunkify=False)
    logger.info('Writing train & test to TFRecord done.')


def read_norb_tfrecord(filenames):
    """Credit: http: // ycszen.github.io / 2016 / 08 / 17 / TensorFlow高效读取数据/
       Original Version: Ycszen-物语
    """

    # 根据文件名生成一个队列
    assert isinstance(filenames, list)

    filename_queue = tf.train.string_input_producer(filenames)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)  # 返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                       })
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [96, 96, 1])
    img = tf.cast(img, tf.float32)  # * (1. / 255) # left normalization to init batch norm layer
    label = tf.cast(features['label'], tf.int32)
    label = tf.one_hot(label, 5, dtype=tf.int32)
    return img, label


def test(is_train: True):
    """Instruction on how to read data from tfrecord"""

    # 1. use regular expression to find all files we want
    import re
    if is_train:
        CHUNK_RE = re.compile(r"train\d+\.tfrecord")
    else:
        CHUNK_RE = re.compile(r"test\d+\.tfrecord")

    processed_dir = './data'
    # 1. parse them into a list of file name
    chunk_files = [os.path.join(processed_dir, fname)
                   for fname in os.listdir(processed_dir)
                   if CHUNK_RE.match(fname)]
    # 2. pass argument into read method
    image, label = read_norb_tfrecord(chunk_files)
    logger.debug('Test read tf record Succeed')


parser = argparse.ArgumentParser()
argh.add_commands(parser, [tfrecord, test])

if __name__ == "__main__":
    argh.dispatch(parser)
