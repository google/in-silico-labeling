# Copyright 2017 Google Inc.
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
# ==============================================================================
"""Tools for reading Seeing More data from disk and preparing it for Brain."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os
import re

import numpy as np
import tensorflow as tf
from typing import Dict, List, NamedTuple, Optional, Tuple, Union

# pylint: disable=g-bad-import-order
from isl import ops
from isl import tensorcheck
from isl import util

gfile = tf.gfile
lt = tf.contrib.labeled_tensor
logging = tf.logging

# pylint: disable=invalid-name
ImageMetadata = NamedTuple('ImageMetadata', [
    ('z', Union[int, str]),
    ('channel', str),
])

# pylint: enable=invalid-name


def parse_image_path(path: str) -> ImageMetadata:
  """Parses an image path into an ImageMetadata."""
  z_pattern = re.compile(r'.*z_depth-(\d+),.*')
  if 'depth_computation-MAXPROJECT' in path:
    z = 'MAXPROJECT'
  else:
    match = z_pattern.match(path)
    if not match:
      raise ValueError('Failed to match z in path: %s', path)
    z = int(match.group(1))

  channel_pattern = re.compile(r'.*channel-(\w+),.*')
  match = channel_pattern.match(path)
  if not match:
    raise ValueError('Failed to match channel in path: %s', path)
  channel = match.group(1)

  return ImageMetadata(z, channel)


def load_image_set(directory: str) -> Dict[ImageMetadata, np.ndarray]:
  """Reads all the PNGs in the given directory."""
  logging.info('Reading image set in: %s', directory)
  image_set = {}
  for f in gfile.ListDirectory(directory):
    path = os.path.join(directory, f)
    if gfile.Exists(path) and os.path.splitext(
        f)[1] == '.png' and 'PREDICTED' not in f:
      image_set[parse_image_path(path)] = util.read_image(path)

  return image_set


def num_z_values(ims: List[ImageMetadata]) -> int:
  """The number of distinct z-values in a set of ImageMetadatas."""
  z_values = []
  for im in ims:
    if isinstance(im.z, int):
      z_values.append(im.z)

  num_z = len(set(z_values))

  if sorted(list(set(z_values))) != list(range(num_z)):
    raise ValueError('Invalid z values: %r', sorted(list(set(z_values))))

  return num_z


def get_image_by_attributes(images: Dict[ImageMetadata, np.ndarray],
                            z: Union[float, str],
                            channel: str) -> Tuple[np.ndarray, np.ndarray]:
  """Returns the image with the given attributes."""
  if isinstance(z, float):
    assert z >= 0.0
    assert z <= 1.0
    z = int(round((num_z_values(images.keys()) - 1.0) * z))

  black_image = np.zeros_like(list(images.values())[0])

  im = ImageMetadata(z, channel)
  if im in images:
    logging.info('Found image matching %r, %r', z, channel)
    return images[im], np.ones_like(images[im])
  else:
    logging.info('Did not find image matching %r, %r', z, channel)
    return black_image, black_image


def tensor_from_array(image: np.ndarray) -> tf.Tensor:
  """Create a tensor from an array without saving the data in the Graph."""
  tensor = tf.py_func(lambda: image, [], tf.float32)
  num_rows, num_columns = image.shape
  tensor.set_shape([num_rows, num_columns])
  return tensor


def image_set_to_tensor(images: Dict[ImageMetadata, np.ndarray],
                        z_values: List[Union[float, str]],
                        channel_values: List[str],
                        row_start: tf.Tensor,
                        column_start: tf.Tensor,
                        crop_size: Optional[int],
                        name: Optional[str] = None) -> lt.LabeledTensor:
  """Converts a set of images into a LabeledTensor.

  Args:
    images: A map from paths to loaded images.
    z_values: See decode_seeing_more_single.
    channel_values: See decode_seeing_more_single.
    row_start: Where to start the crop, or None for random.
    column_start: Where to start the crop, or None for random.
    crop_size: The size of the crop.
    name: Optional op name.

  Returns:
    A LabeledTensor with the images.

  Raises:
    ValueError: If zero z or channel values are supplied.
  """
  with tf.name_scope(name, 'load_image_set_as_tensor', []) as scope:

    def crop(image: np.ndarray) -> tf.Tensor:
      tensor = tensor_from_array(image)
      return tensor[row_start:row_start + crop_size, column_start:
                    column_start + crop_size]

    image_list = []
    for z in z_values:
      for channel in channel_values:
        image, mask = get_image_by_attributes(images, z, channel)
        logging.info('z: %r, c: %r, mask: False, min: %r, max: %r', z, channel,
                     image.min(), image.max())
        logging.info('z: %r, c: %r, mask: True, min: %r, max: %r', z, channel,
                     mask.min(), mask.max())
        image_list.extend([crop(image), crop(mask)])
    if not image_list:
      raise ValueError('Must supply more than zero z and channel values')

    mask_values = [False, True]
    image_op = tf.reshape(
        tf.stack(image_list, axis=2), [
            1, crop_size, crop_size,
            len(z_values),
            len(channel_values),
            len(mask_values)
        ],
        name=scope)

    axes = [('batch', 1), 'row', 'column', ('z', z_values),
            ('channel', channel_values), ('mask', mask_values)]
    return lt.LabeledTensor(image_op, axes)


def read_serialized_example(
    shard_paths: List[str],
    is_recordio: bool,
    bp: util.BatchParameters,
    is_deterministic: bool,
    name: str = None,
) -> lt.LabeledTensor:
  """Read serialized tf.Example protos from a RecordIO or SSTable.

  Args:
    shard_paths: The list of file shards from which to read.
    is_recordio: Whether the data is in the RecordIO format.
      If not, it must be in an SSTable.
    bp: Batch parameters.
    is_deterministic: Whether to use a fixed random seed.
    name: An optional op name.

  Returns:
    A tensor of shape [batch_size] containing serialized tf.Example protos.
  """
  with tf.name_scope(name, 'read_serialized_example', []) as scope:
    if is_deterministic:
      seed = 0
    else:
      seed = None
    filename_op = tf.train.string_input_producer(shard_paths, seed=seed)

    if is_recordio:
      reader = tf.TFRecordReader()
    else:
      raise NotImplementedError('Only TFRecords are currently supported.')
    _, serialized_example_op = reader.read(filename_op)

    serialized_example_lt = lt.LabeledTensor(serialized_example_op, [])

    [batch_lt] = lt.shuffle_batch(
        [serialized_example_lt],
        batch_size=bp.size,
        enqueue_many=False,
        min_after_dequeue=bp.capacity // 2,
        num_threads=bp.num_threads,
        capacity=bp.capacity,
        seed=seed)

    # C++ entry point.
    with tf.name_scope(''):
      batch_lt = lt.identity(batch_lt, name='entry_point_serialized_example')
    return lt.identity(batch_lt, name=scope)


def decode_single(
    z_values: List[Union[float, str]],
    channel_values: List[str],
    mask_values: List[bool],
    pad_size: int,
    crop_size: int,
    serialized_example_lt: lt.LabeledTensor,
    name: str = None,
) -> lt.LabeledTensor:
  """Decode a single serialized tf.Example.

  Args:
    z_values: The list of z values to extract.
    channel_values: The list of channels to extract.
    mask_values: The subset of [original, mask] images to extract.
      For example, if this is [False, True], extract the original and
      mask images.
    pad_size: Size of zeros border to add to images before random cropping.
    crop_size: The size (width and height) of the crop.
    serialized_example_lt: The serialized example, of shape [1].
      We assume the example contains serialized 16-bit grayscale PNGS.
    name: Optional op name.

  Returns:
    A dictionary containing an image for each element in the direct product
    z_values x channel_values x mask_values.
    Each image has shape [1, image_size, image_size], and images missing
    from the tf.Example are black.
    Image pixels are floats in the range [0.0, 1.0].
  """
  with tf.name_scope(name, 'decode_single', [serialized_example_lt]) as scope:
    features = {'TRAIN_TUPLE': tf.FixedLenFeature([], dtype=tf.string)}
    encoded_train_tuple_op = tf.reshape(
        tf.parse_example(tf.reshape(serialized_example_lt, [1]),
                         features)['TRAIN_TUPLE'], [])

    encoded_image_op = tf.reshape(
        ops.train_tuple_to_images([str(z) for z in z_values], channel_values,
                                  mask_values, encoded_train_tuple_op),
        [len(z_values) * len(channel_values) * len(mask_values)])

    images_op = tf.map_fn(
        functools.partial(tf.image.decode_png, channels=1, dtype=tf.uint16),
        encoded_image_op,
        dtype=tf.uint16)
    images_op = tf.squeeze(images_op, [3])
    # This has shape row x column x (z * channel * mask).
    images_op = tf.transpose(images_op, [1, 2, 0])

    if pad_size > 0:
      images_op = tf.pad(
          images_op, [[pad_size, pad_size], [pad_size, pad_size], [0, 0]],
          mode='CONSTANT')

    images_op = tf.random_crop(
        images_op, [crop_size, crop_size,
                    images_op.shape_as_list()[2]], seed=0)
    images_op = tensorcheck.shape_unlabeled(images_op)
    # We assume 16-bit images.
    images_op = tf.to_float(images_op) / np.iinfo(np.uint16).max

    images_op = tensorcheck.bounds_unlabeled(0.0, 1.0, images_op)

    images_op = tf.reshape(
        images_op, [
            crop_size, crop_size,
            len(z_values),
            len(channel_values),
            len(mask_values)
        ],
        name=scope)

    axes = [('row', crop_size), ('column', crop_size), ('z', z_values),
            ('channel', channel_values), ('mask', mask_values)]
    return lt.LabeledTensor(images_op, axes)


def decode_seeing_more(
    z_values: List[Union[float, str]],
    channel_values: List[str],
    mask_values: List[bool],
    pad_size: int,
    crop_size: int,
    serialized_example_lt: lt.LabeledTensor,
    name: str = None,
) -> lt.LabeledTensor:
  """A batch version of decode_single."""
  with tf.name_scope(name, 'decode_seeing_more',
                     [serialized_example_lt]) as scope:
    serialized_example_lt = lt.transpose(serialized_example_lt, ['batch'])

    def decode(t):
      return decode_single(z_values, channel_values, mask_values, pad_size,
                           crop_size, t)

    return lt.map_fn(decode, serialized_example_lt, name=scope)


def add_neurite_confocal(
    target_lt: lt.LabeledTensor,
    name: str = None,
) -> lt.LabeledTensor:
  """Add the synthetic target neurite channel.

  Args:
    target_lt: Input target tensor.
    name: Optional op name.

  Returns:
    The target tensor with a new 'NEURITE_CONFOCAL' channel, which is the
    average of the NFH_CONFOCAL and MAP2_CONFOCAL channels.
  """
  with tf.name_scope(name, 'add_neurite_confocal', [target_lt]) as scope:
    target_lt = lt.transpose(target_lt, util.CANONICAL_AXIS_ORDER)
    neurite_lts = []
    for m in [False, True]:
      nfh_lt = lt.select(target_lt, {
          'channel': util.slice_1('NFH_CONFOCAL'),
          'mask': util.slice_1(m)
      })
      nfh_lt = lt.reshape(nfh_lt, ['channel'],
                          [('channel', ['NEURITE_CONFOCAL'])])
      map2_lt = lt.select(target_lt, {
          'channel': util.slice_1('MAP2_CONFOCAL'),
          'mask': util.slice_1(m)
      })
      map2_lt = lt.reshape(map2_lt, ['channel'],
                           [('channel', ['NEURITE_CONFOCAL'])])

      if not m:
        # This corresponds to a logical OR.
        neurite_lts.append((nfh_lt + map2_lt) / 2.0)
      else:
        # The combined mask is the geometric mean of the original masks.
        # This corresponds to a logical AND.
        neurite_lts.append(lt.pow(nfh_lt * map2_lt, 0.5))

    neurite_lt = lt.concat(neurite_lts, 'mask')
    return lt.concat([target_lt, neurite_lt], 'channel', name=scope)


def add_synthetic_channels(
    target_lt: lt.LabeledTensor,
    name: str = None,
) -> lt.LabeledTensor:
  """Adds all synthetic channels.

  We're currently just adding a synthetic neurite channel.

  Args:
    target_lt: The input target tensor.
    name: Optional op name.

  Returns:
    The input target tensor, with additional synthetic channels.
  """
  with tf.name_scope(name, 'add_synthetic_channels',
                     [target_lt.tensor]) as scope:
    if 'NFH_CONFOCAL' in target_lt.axes['channel'].labels and 'MAP2_CONFOCAL' in target_lt.axes['channel'].labels:
      return add_neurite_confocal(target_lt, name=scope)
    else:
      return lt.identity(target_lt, name=scope)


# Parameters for reading from a folder of PNGs.
# pylint: disable=invalid-name
ReadPNGsParameters = NamedTuple('ReadPNGsParameters', [
    ('directory', str),
    ('row_start', Optional[int]),
    ('column_start', Optional[int]),
    ('crop_size', int),
])
# pylint: enable=invalid-name

# Parameters for reading from a RecordIO or SSTable.
# pylint: disable=invalid-name
ReadTableParameters = NamedTuple('ReadTableParameters', [
    ('shard_paths', List[str]),
    ('is_recordio', bool),
    ('bp', util.BatchParameters),
    ('is_deterministic', bool),
    ('pad_size', int),
    ('crop_size', int),
])
# pylint: enable=invalid-name

# Convience class for data parameters.
# pylint: disable=invalid-name
DataParameters = NamedTuple('DataParameters', [
    ('io_parameters', Union[ReadPNGsParameters, ReadTableParameters]),
    ('input_z_values', List[Union[float, str]]),
    ('input_channel_values', List[str]),
    ('target_z_values', List[Union[float, str]]),
    ('target_channel_values', List[str]),
])

# pylint: enable=invalid-name


def cropped_input_and_target(
    dp: DataParameters,
    name: str = None,
) -> Tuple[lt.LabeledTensor, lt.LabeledTensor]:
  """Reads and crops input images."""
  with tf.name_scope(name, 'cropped_input_and_target', []) as scope:
    if isinstance(dp.io_parameters, ReadPNGsParameters):
      image_set = load_image_set(dp.io_parameters.directory)
      num_rows, num_columns = list(image_set.values())[0].shape
      assert num_rows >= dp.io_parameters.crop_size, (
          'Crop size must not be larger than the '
          'number of image rows.')
      assert num_columns >= dp.io_parameters.crop_size, (
          'Crop size must not be larger than the '
          'number of image columns.')

      if dp.io_parameters.row_start is None:
        row_start = tf.random_uniform(
            [], 0, num_rows - dp.io_parameters.crop_size + 1, dtype=tf.int32)
      else:
        row_start = dp.io_parameters.row_start

      if dp.io_parameters.column_start is None:
        column_start = tf.random_uniform(
            [], 0, num_columns - dp.io_parameters.crop_size + 1, dtype=tf.int32)
      else:
        column_start = dp.io_parameters.column_start

      input_lt = image_set_to_tensor(image_set, dp.input_z_values,
                                     dp.input_channel_values, row_start,
                                     column_start, dp.io_parameters.crop_size)

      target_lt = image_set_to_tensor(image_set, dp.target_z_values,
                                      dp.target_channel_values, row_start,
                                      column_start, dp.io_parameters.crop_size)
      target_lt = add_synthetic_channels(target_lt, name=scope + 'target')
    else:
      serialized_example_lt = read_serialized_example(
          dp.io_parameters.shard_paths, dp.io_parameters.is_recordio,
          dp.io_parameters.bp, dp.io_parameters.is_deterministic)

      input_lt = decode_seeing_more(dp.input_z_values, dp.input_channel_values,
                                    [False, True], dp.io_parameters.pad_size,
                                    dp.io_parameters.crop_size,
                                    serialized_example_lt)

      target_lt = decode_seeing_more(
          dp.target_z_values, dp.target_channel_values, [False, True],
          dp.io_parameters.pad_size, dp.io_parameters.crop_size,
          serialized_example_lt)
      target_lt = add_synthetic_channels(target_lt, name=scope + 'target')

    input_lt = tensorcheck.bounds(0.0, 1.0, input_lt, name='input_pre_sum')
    # We assume exactly one input channel is provided, which we call
    # "TRANSMISSION".
    input_lt = lt.reduce_sum(
        input_lt, [('channel', 'TRANSMISSION')], name=scope + 'input')
    input_lt = tensorcheck.bounds(0.0, 1.0, input_lt, name='input_post_sum')

    target_lt = tensorcheck.bounds(0.0, 1.0, target_lt, name='target')

    logging.info('input: %r', input_lt)
    logging.info('target: %r', target_lt)

    return input_lt, target_lt
