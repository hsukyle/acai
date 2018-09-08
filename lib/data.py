# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Input data for GANs.

This module provides the input images.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import glob
import math
import os
import random

import numpy as np
import tensorflow as tf
import ipdb

_DATA_CACHE = None

DATA_DIR = './data'

class DataSet(object):

    def __init__(self, name, train, val, test, train_once, height, width, colors,
                 nclass):
        self.name = name
        self.train = train
        self.train_once = train_once
        self.val = val
        self.test = test
        self.height = height
        self.width = width
        self.colors = colors
        self.nclass = nclass


def get_dataset(dataset_name, params):
    if dataset_name == 'mnist32':   # one-hot labels for downstream evaluation, etc. during AE training
        train, height, width, colors = _DATASETS[dataset_name + '_train'](
            batch_size=params['batch_size'])
        val = _DATASETS[dataset_name + '_val'](batch_size=1)[0]
        test = _DATASETS[dataset_name + '_test'](batch_size=1)[0]
        train = train.map(lambda v: dict(x=v['x'],
                                         label=tf.one_hot(v['label'],
                                                          _NCLASS[dataset_name]),
                                         x_orig=v['x_orig']))
        val = val.map(lambda v: dict(x=v['x'], label=tf.one_hot(v['label'], _NCLASS[dataset_name]),
                                     x_orig=v['x_orig']))
        test = test.map(lambda v: dict(x=v['x'],
                                       label=tf.one_hot(v['label'],
                                                        _NCLASS[dataset_name]),
                                       x_orig=v['x_orig']))
        if dataset_name + '_train_once' in _DATASETS:
            train_once = _DATASETS[dataset_name + '_train_once'](batch_size=1)[0]
            train_once = train_once.map(lambda v: dict(
                x=v['x'], label=tf.one_hot(v['label'], _NCLASS[dataset_name]), x_orig=v['x_orig']))
        else:
            train_once = None
        return DataSet(dataset_name, train, val, test, train_once, height, width,
                       colors, _NCLASS[dataset_name])
    else:   # labels are information we'd like to keep track of, not used
        train, height, width, colors = _DATASETS[dataset_name + '_train'](
            batch_size=params['batch_size'])
        val = _DATASETS[dataset_name + '_val'](batch_size=1)[0]
        test = _DATASETS[dataset_name + '_test'](batch_size=1)[0]
        train = train.map(lambda v: dict(x=v['x'], label=v['label'], x_orig=v['x_orig']))
        val = val.map(lambda v: dict(x=v['x'], label=v['label'], x_orig=v['x_orig']))
        test = test.map(lambda v: dict(x=v['x'], label=v['label'], x_orig=v['x_orig']))
        if dataset_name + '_train_once' in _DATASETS:
            train_once = _DATASETS[dataset_name + '_train_once'](batch_size=1)[0]
            train_once = train_once.map(lambda v: dict(x=v['x'], label=v['label'], x_orig=v['x_orig']))
        else:
            train_once = None

        return DataSet(dataset_name, train, val, test, train_once, height, width,
                       colors, _NCLASS[dataset_name])


def draw_line(angle, height, width, w=2.):
    m = np.zeros((height, width, 1))
    x0 = height*0.5
    y0 = width*0.5
    x1 = x0 + (x0 - 1) * math.cos(-angle)
    y1 = y0 + (y0 - 1) * math.sin(-angle)
    flip = False
    if abs(y0 - y1) < abs(x0 - x1):
        x0, x1, y0, y1 = y0, y1, x0, x1
        flip = True
    if y1 < y0:
        x0, x1, y0, y1 = x1, x0, y1, y0
    x0, x1 = x0 - w / 2, x1 - w / 2
    dx = x1 - x0
    dy = y1 - y0
    ds = dx / dy if dy != 0 else 0
    yi = int(math.ceil(y0)), int(y1)
    points = []
    for y in range(int(y0), int(math.ceil(y1))):
        if y < yi[0]:
            weight = yi[0] - y0
        elif y > yi[1]:
            weight = y1 - yi[1]
        else:
            weight = 1
        xs = x0 + (y - y0 - .5) * ds
        xe = xs + w
        xi = int(math.ceil(xs)), int(xe)
        if xi[0] != xi[1]:
            points.append((y, slice(xi[0], xi[1]), weight))
        if xi[0] != xs:
            points.append((y, int(xs), weight * (xi[0] - xs)))
        if xi[1] != xe:
            points.append((y, xi[1], weight * (xe - xi[1])))
    if flip:
        points = [(x, y, z) for y, x, z in points]
    for y, x, z in points:
        m[y, x] += 2 * z
    m -= 1
    m = m.clip(-1, 1)
    return m


def input_lines(batch_size, size=(32, 32, 1), limit=None):
    h, w, c = size

    def gen():
        count = 0
        while limit is None or count < limit:
            angle = 2 * random.random() * math.pi
            m = draw_line(angle, h, w)
            label = int(10 * angle / (2 * math.pi - 1e-6))
            count += 1
            yield m, label

    ds = tf.data.Dataset.from_generator(gen, (tf.float32, tf.int64),
                                        (size, tuple()))
    ds = ds.map(lambda x, y: dict(x=x, label=y))
    ds = ds.batch(batch_size)
    return ds, size[0], size[1], size[2]


def _parser_all(serialized_example):
    features = tf.parse_single_example(
        serialized_example,
        features={'image': tf.FixedLenFeature([], tf.string),
                  'label': tf.FixedLenFeature([], tf.int64)})
    image_orig = tf.image.decode_image(features['image'])
    image = tf.cast(image_orig, tf.float32) * (2.0 / 255) - 1.0
    label = features['label']
    return image, label, image_orig


def input_fn_record(record_parse_fn,
                    filenames,
                    batch_size,
                    size=(32, 32, 3),
                    pad=(0, 0),
                    crop=(0, 0),
                    resize=(32, 32),
                    shuffle=1024,
                    repeat=True,
                    random_flip_x=False,
                    random_shift_x=0,
                    random_shift_y=0,
                    limit=None,
                    random_crop_to_resize=False,
                    grayscale=False):
    """Creates a Dataset pipeline for tfrecord files.

    Args:
    record_parse_fn: function, used to parse a record entry.
    filenames: list of filenames of the tfrecords.
    batch_size: int, batch size.
    size: tuple (HWC) containing the expected image shape.
    pad: tuple (HW) containing how much to pad y and x axis on each size.
    crop: tuple (HW) containing how much to crop y and x axis.
    resize: tuple (HW) containing the desired image shape.
    shuffle: int, the size of the shuffle buffer.
    repeat: bool, whether the dataset repeats itself.
    random_flip_x: bool, whether to random flip the x-axis.
    random_shift_x: int, amount of random horizontal shift.
    random_shift_y: int, amount of random vertical shift.
    limit: int, the number of samples to drop (<0) or to take (>0)..

    Returns:
    Dataset iterator and 3 ints (height, width, colors).
    """

    def random_shift(v):
        if random_shift_y:
            v = tf.concat([v[-random_shift_y:], v, v[:random_shift_y]], 0)
        if random_shift_x:
            v = tf.concat([v[:, -random_shift_x:], v, v[:, :random_shift_x]],
                          1)
        return tf.random_crop(v, [resize[0], resize[1], size[2]])

    filenames = sum([glob.glob(x) for x in filenames], [])
    if not filenames:
        raise ValueError('Empty dataset, did you mount gcsfuse bucket?')
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(record_parse_fn, max(4, batch_size // 4))
    if grayscale:
        dataset = dataset.map(
            lambda x, y, x_orig: (tf.image.rgb_to_grayscale(x), y, x_orig))
    if limit is not None:
        if limit > 0:
            dataset = dataset.take(limit)
        elif limit < 0:
            dataset = dataset.skip(-limit)
    if repeat:
        dataset = dataset.repeat()
    delta = [0, 0]
    if sum(crop):
        dataset = dataset.map(
            lambda x, y, x_orig: (x[crop[0]:-crop[0], crop[1]:-crop[1]], y, x_orig))
        delta[0] -= 2 * crop[0]
        delta[1] -= 2 * crop[1]
    if sum(pad):
        padding = [[pad[0]] * 2, [pad[1]] * 2, [0] * 2]
        dataset = dataset.map(
            lambda x, y, x_orig: (tf.pad(x, padding, constant_values=-1.), y, x_orig))
        delta[0] += 2 * crop[0]
        delta[1] += 2 * crop[1]

    if resize[0] - delta[0] != size[0] or resize[1] - delta[1] != size[1]:
        if random_crop_to_resize:
            dataset = dataset.map(
                lambda x, y, x_orig: (tf.random_crop(x, (resize[0], resize[1], size[-1])), y, x_orig), 4)
        else:
            dataset = dataset.map(
                lambda x, y, x_orig: (tf.image.resize_bicubic([x], list(resize))[0], y, x_orig), 4)
    if shuffle:
        dataset = dataset.shuffle(shuffle)
    if random_flip_x:
        dataset = dataset.map(
            lambda x, y, x_orig: (tf.image.random_flip_left_right(x), y, x_orig), 4)
    if random_shift_x or random_shift_y:
        dataset = dataset.map(lambda x, y, x_orig: (random_shift(x), y, x_orig), 4)
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(
        lambda x, y, x_orig: dict(
            x=tf.reshape(x, [batch_size] + list(resize) + list(size[-1:])),
            label=y, x_orig=x_orig))
    dataset = dataset.prefetch(4)  # Prefetch a few batches.
    return dataset, resize[0] or size[0], resize[1] or size[1], size[2]


_NCLASS = {
    'celeba32': 1,
    'cifar10': 10,
    'lines32': 10,
    'mnist32': 10,
    'svhn32': 10,
    'omniglot32': -1,
    'miniimagenet64': -1,
    'miniimagenet32': -1,
    'vizdoom': -1,
    'celeba64': -1,
    'miniimagenetgray64': -1,
}

_DATASETS = {
    'celeba64_train':
        functools.partial(
            input_fn_record,
            _parser_all,
            [os.path.join(DATA_DIR, 'celeba-train.tfrecord')],
            size=(84, 84, 3),
            resize=(64, 64)),
    'celeba64_test':
        functools.partial(
            input_fn_record,
            _parser_all,
            [os.path.join(DATA_DIR, 'celeba-test.tfrecord')],
            shuffle=False,
            repeat=False,
            size=(84, 84, 3),
            resize=(64, 64)),
    'celeba64_val':
        functools.partial(
            input_fn_record,
            _parser_all,
            [os.path.join(DATA_DIR, 'celeba-val.tfrecord')],
            shuffle=False,
            repeat=False,
            size=(84, 84, 3),
            resize=(64, 64)),
    'celeba64_train_once':
        functools.partial(
            input_fn_record,
            _parser_all,
            [os.path.join(DATA_DIR, 'celeba-train.tfrecord')],
            shuffle=False,
            repeat=False,
            size=(84, 84, 3),
            resize=(64, 64)),
    'vizdoom_train':
        functools.partial(
            input_fn_record,
            _parser_all,
            [os.path.join(DATA_DIR, 'vizdoom-train.tfrecord')],
            size=(100, 130, 3),
            resize=(96, 128)),
    'vizdoom_val':
        functools.partial(
            input_fn_record,
            _parser_all,
            [os.path.join(DATA_DIR, 'vizdoom-train.tfrecord')],
            size=(100, 130, 3),
            resize = (96, 128)),
    'vizdoom_test':
        functools.partial(
            input_fn_record,
            _parser_all,
            [os.path.join(DATA_DIR, 'vizdoom-train.tfrecord')],
            size=(100, 130, 3),
            resize=(96, 128)),
    'celeba32_train':
        functools.partial(
            input_fn_record,
            _parser_all,
            [os.path.join(DATA_DIR, 'celeba-train.tfrecord')],
            size=(218, 178, 3),
            crop=(36, 16),
            resize=(32, 32)),
    'celeba32_test':
        functools.partial(
            input_fn_record,
            _parser_all,
            [os.path.join(DATA_DIR, 'celeba-train.tfrecord')],
            shuffle=False,
            repeat=False,
            size=(218, 178, 3),
            crop=(36, 16),
            resize=(32, 32)),
    'cifar10_train_once':
        functools.partial(
            input_fn_record,
            _parser_all,
            [os.path.join(DATA_DIR, 'cifar10-train.tfrecord')],
            shuffle=False,
            repeat=False,
            size=(32, 32, 3),
            resize=(32, 32)),
    'cifar10_train':
        functools.partial(
            input_fn_record,
            _parser_all,
            [os.path.join(DATA_DIR, 'cifar10-train.tfrecord')],
            size=(32, 32, 3),
            resize=(32, 32)),
    'cifar10_test':
        functools.partial(
            input_fn_record,
            _parser_all,
            [os.path.join(DATA_DIR, 'cifar10-test.tfrecord')],
            shuffle=False,
            repeat=False,
            size=(32, 32, 3),
            resize=(32, 32)),
    'cifar10_val':
        functools.partial(
            input_fn_record,
            _parser_all,
            [os.path.join(DATA_DIR, 'cifar10-test.tfrecord')],
            shuffle=False,
            repeat=False,
            size=(32, 32, 3),
            resize=(32, 32)),
    'miniimagenet32_train':
        functools.partial(
            input_fn_record,
            _parser_all,
            [os.path.join(DATA_DIR, 'miniimagenet-train.tfrecord')],
            size=(84, 84, 3),
            resize=(32, 32),
            random_crop_to_resize=True),
    'miniimagenet32_test':
        functools.partial(
            input_fn_record,
            _parser_all,
            [os.path.join(DATA_DIR, 'miniimagenet-test.tfrecord')],
            shuffle=False,
            repeat=False,
            size=(84, 84, 3),
            resize=(32, 32)),
    'miniimagenet32_val':
        functools.partial(
            input_fn_record,
            _parser_all,
            [os.path.join(DATA_DIR, 'miniimagenet-val.tfrecord')],
            shuffle=False,
            repeat=False,
            size=(84, 84, 3),
            resize=(32, 32)),
    'miniimagenet32_train_once':
        functools.partial(
            input_fn_record,
            _parser_all,
            [os.path.join(DATA_DIR, 'miniimagenet-train.tfrecord')],
            shuffle=False,
            repeat=False,
            size=(84, 84, 3),
            resize=(32, 32)),
    'miniimagenetgray64_train':
        functools.partial(
            input_fn_record,
            _parser_all,
            [os.path.join(DATA_DIR, 'miniimagenet-train.tfrecord')],
            size=(84, 84, 3),
            resize=(64, 64),
            random_crop_to_resize=True,
            grayscale=True),
    'miniimagenetgray64_test':
        functools.partial(
            input_fn_record,
            _parser_all,
            [os.path.join(DATA_DIR, 'miniimagenet-test.tfrecord')],
            shuffle=False,
            repeat=False,
            size=(84, 84, 3),
            resize=(64, 64),
            grayscale=True),
    'miniimagenetgray64_val':
        functools.partial(
            input_fn_record,
            _parser_all,
            [os.path.join(DATA_DIR, 'miniimagenet-val.tfrecord')],
            shuffle=False,
            repeat=False,
            size=(84, 84, 3),
            resize=(64, 64),
            grayscale=True),
    'miniimagenetgray64_train_once':
        functools.partial(
            input_fn_record,
            _parser_all,
            [os.path.join(DATA_DIR, 'miniimagenet-train.tfrecord')],
            shuffle=False,
            repeat=False,
            size=(84, 84, 3),
            resize=(64, 64),
            grayscale=True),
    'miniimagenet64_train':
        functools.partial(
            input_fn_record,
            _parser_all,
            [os.path.join(DATA_DIR, 'miniimagenet-train.tfrecord')],
            size=(84, 84, 3),
            resize=(64, 64),
            random_crop_to_resize=True),
    'miniimagenet64_test':
        functools.partial(
            input_fn_record,
            _parser_all,
            [os.path.join(DATA_DIR, 'miniimagenet-test.tfrecord')],
            shuffle=False,
            repeat=False,
            size=(84, 84, 3),
            resize=(64, 64)),
    'miniimagenet64_val':
        functools.partial(
            input_fn_record,
            _parser_all,
            [os.path.join(DATA_DIR, 'miniimagenet-val.tfrecord')],
            shuffle=False,
            repeat=False,
            size=(84, 84, 3),
            resize=(64, 64)),
    'miniimagenet64_train_once':
        functools.partial(
            input_fn_record,
            _parser_all,
            [os.path.join(DATA_DIR, 'miniimagenet-train.tfrecord')],
            shuffle=False,
            repeat=False,
            size=(84, 84, 3),
            resize=(64, 64)),
    'omniglot32_train_once':
        functools.partial(
            input_fn_record,
            _parser_all,
            [os.path.join(DATA_DIR, 'omniglot-train.tfrecord')],
            repeat=False,
            shuffle=False,
            size=(28, 28, 1),
            pad=(2, 2),
            resize=(32, 32)),
    'omniglot32_train':
        functools.partial(
            input_fn_record,
            _parser_all,
            [os.path.join(DATA_DIR, 'omniglot-train.tfrecord')],
            size=(28, 28, 1),
            pad=(2, 2),
            resize=(32, 32)),
    'omniglot32_test':
        functools.partial(
            input_fn_record,
            _parser_all,
            [os.path.join(DATA_DIR, 'omniglot-test.tfrecord')],
            shuffle=False,
            repeat=False,
            size=(28, 28, 1),
            pad=(2, 2),
            resize=(32, 32)),
    'omniglot32_val':
        functools.partial(
            input_fn_record,
            _parser_all,
            [os.path.join(DATA_DIR, 'omniglot-val.tfrecord')],
            shuffle=False,
            repeat=False,
            size=(28, 28, 1),
            pad=(2, 2),
            resize=(32, 32)),
    'mnist32_train_once':
        functools.partial(
            input_fn_record,
            _parser_all,
            [os.path.join(DATA_DIR, 'mnist-train.tfrecord')],
            repeat=False,
            shuffle=False,
            size=(28, 28, 1),
            pad=(2, 2),
            resize=(32, 32)),
    'mnist32_train':
        functools.partial(
            input_fn_record,
            _parser_all,
            [os.path.join(DATA_DIR, 'mnist-train.tfrecord')],
            size=(28, 28, 1),
            pad=(2, 2),
            resize=(32, 32)),
    'mnist32_test':
        functools.partial(
            input_fn_record,
            _parser_all,
            [os.path.join(DATA_DIR, 'mnist-test.tfrecord')],
            shuffle=False,
            repeat=False,
            size=(28, 28, 1),
            pad=(2, 2),
            resize=(32, 32)),
    'mnist32_val':
        functools.partial(
            input_fn_record,
            _parser_all,
            [os.path.join(DATA_DIR, 'mnist-val.tfrecord')],
            shuffle=False,
            repeat=False,
            size=(28, 28, 1),
            pad=(2, 2),
            resize=(32, 32)),
    'svhn32_train_once':
        functools.partial(
            input_fn_record,
            _parser_all,
            [os.path.join(DATA_DIR, 'svhn-train.tfrecord'),
             os.path.join(DATA_DIR, 'svhn-extra.tfrecord')],
            repeat=False,
            shuffle=False,
            size=(32, 32, 3)),
    'svhn32_train':
        functools.partial(
            input_fn_record,
            _parser_all,
            [os.path.join(DATA_DIR, 'svhn-train.tfrecord'),
             os.path.join(DATA_DIR, 'svhn-extra.tfrecord')],
            size=(32, 32, 3)),
    'svhn32_test':
        functools.partial(
            input_fn_record,
            _parser_all,
            [os.path.join(DATA_DIR, 'svhn-test.tfrecord')],
            shuffle=False,
            repeat=False,
            size=(32, 32, 3)),
    'lines32_train': functools.partial(input_lines, size=(32, 32, 1)),
    'lines32_test': functools.partial(input_lines, limit=5000,
                                         size=(32, 32, 1)),
}
