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
"""Utility functions and globals."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os

# pylint: disable=g-bad-import-order
import numpy as np
import tensorflow as tf
from typing import Callable, Dict, List, Tuple, Union
import cv2

gfile = tf.gfile
logging = tf.logging
lt = tf.contrib.labeled_tensor
slim = tf.contrib.slim

# The standard axis order for Seeing More ground truth tensors.
# These represent data read from disk, before going through the model.
CANONICAL_AXIS_ORDER = ['batch', 'row', 'column', 'z', 'channel', 'mask']

# The standard axis order for Seeing More prediction tensors.
# These represent data output from the model; the values are either logits
# or probabilities.
CANONICAL_PREDICTION_AXIS_ORDER = [
    'batch', 'row', 'column', 'z', 'channel', 'class'
]

# The standard axis order for Seeing More summary statistic tensors.
# The are currently only created by ops.distribution_statistics.
CANONICAL_STATISTIC_AXIS_ORDER = [
    'batch', 'row', 'column', 'z', 'channel', 'statistic'
]


def read_image(path: str) -> np.ndarray:
  """Reads a 16-bit grayscale image and converts to floating point."""
  logging.info('Reading image: %s', path)
  image = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
  assert image is not None
  assert len(image.shape) == 2, image.shape
  assert image.dtype == np.uint8 or image.dtype == np.uint16

  image = image.astype(np.float32) / np.iinfo(image.dtype).max
  assert image.min() >= 0, image.min()
  assert image.max() <= 1.0, image.max()
  return image


def write_image(path: str, image: np.ndarray):
  """Writes the image to disk."""
  image = (image * np.iinfo(np.uint16).max).astype(np.uint16)
  if len(image.shape) == 3:
    image = np.stack([image[:, :, 2], image[:, :, 1], image[:, :, 0]], axis=2)
  logging.info('Writing image: %s', path)
  cv2.imwrite(path, image)


def image_size(directory: str) -> Tuple[int, int]:
  """Get the dimensions of the images in the directory."""
  png_paths = []
  for f in gfile.ListDirectory(directory):
    path = os.path.join(directory, f)
    if gfile.Exists(path) and os.path.splitext(f)[1] == '.png':
      png_paths.append(os.path.join(directory, f))
  if not png_paths:
    raise ValueError('No pngs in %s', directory)

  image = read_image(png_paths[0])

  return image.shape[0], image.shape[1]


def slice_1(x):
  """Given x return slice(x, x)."""
  return slice(x, x)


class BatchParameters(object):
  """Convenience class for standard batch parameters."""

  def __init__(self, size: int, num_threads: int, capacity: int):
    self.size = size
    self.num_threads = num_threads
    self.capacity = capacity


def onehot(
    num_classes: int,
    labeled_tensor: lt.LabeledTensor,
    name: str = None,
) -> lt.LabeledTensor:
  """Gets the one-hot encoding of rounded values in [0.0, 1.0].

  See slim.one_hot_encoding.

  Args:
    num_classes: The number of classes in the encoding.
    labeled_tensor: The input tensor.
    name: Optional op name.

  Returns:
    A tensor, with the same axes as the input, plus a new axis "class", which
    has size num_classes.
    The classes are computed by dividing the unit interval into num_classes
    bins.
  """
  with tf.name_scope(name, 'onehot', [labeled_tensor]) as scope:
    reshape_op = tf.reshape(labeled_tensor.tensor, [-1])
    categorical_op = tf.to_int64(tf.round(reshape_op * (num_classes - 1)))
    onehot_op = slim.one_hot_encoding(categorical_op, num_classes)
    onehot_op = tf.reshape(
        onehot_op,
        labeled_tensor.tensor.shape.as_list() + [num_classes],
        name=scope)

    axes = list(labeled_tensor.axes.values()) + [('class', num_classes)]
    return lt.LabeledTensor(onehot_op, axes)


def crop_center(
    size: int,
    input_lt: lt.LabeledTensor,
    name: str = None,
) -> lt.LabeledTensor:
  """Center crop the 'row' and 'column' axes.

  Args:
    size: The width and height of the cropped region.
    input_lt: The input tensor.
    name: Optional op name.

  Returns:
    The center cropped tensor.
  """
  with tf.name_scope(name, 'crop_center', [input_lt]) as scope:
    num_rows = len(input_lt.axes['row'])
    num_columns = len(input_lt.axes['column'])

    assert (num_rows - size) % 2 == 0
    assert (num_columns - size) % 2 == 0

    row_offset = (num_rows - size) // 2
    column_offset = (num_columns - size) // 2

    return lt.slice(
        input_lt, {
            'row': slice(row_offset, num_rows - row_offset),
            'column': slice(column_offset, num_columns - column_offset)
        },
        name=scope)


# TODO(ericmc): Remove this when the core graph ops are rewritten to use
# LabeledTensor.
def crop_center_unlabeled(
    size: int,
    input_op: tf.Tensor,
    name: str = None,
) -> tf.Tensor:
  """Applies crop_center to an unlabeled tensor."""
  input_lt = lt.LabeledTensor(input_op, ['batch', 'row', 'column', 'channel'])
  crop_lt = crop_center(size, input_lt, name=name)
  return crop_lt.tensor


def pad_constant(
    labeled_tensor: lt.LabeledTensor,
    paddings: Dict[str, Tuple[object, object]],
    value: Union[int, float],
    name: str = None,
) -> lt.LabeledTensor:
  """Pads a tensor with a constant scalar value.

  See tf.pad and lt.pad.

  Args:
    labeled_tensor: The input tensor.
    paddings: A mapping where the keys are axis names and the values are
      tuples where the first element is the padding to insert at the beginning
      of the axis and the second is the padding to insert at the end of the
      axis.
    value: The scalar value to pad with.
    name: Optional op name.

  Returns:
    A tensor with the indicated axes padded, optionally with those axes extended
    with the provided labels.
  """
  with tf.name_scope(name, 'pad_constant', [labeled_tensor]) as scope:
    # The constant padding value is zero.
    zero_padded_lt = lt.pad(labeled_tensor, paddings, 'CONSTANT')

    # Construct a tensor that has zeros on the interior and value `value` in
    # the padded regions.
    # TODO(ericmc): This should probably be directly supported by
    # core Tensorflow op.
    scalar_lt = lt.ones_like(labeled_tensor) * (-value)
    scalar_lt = lt.pad(scalar_lt, paddings, 'CONSTANT')
    scalar_lt += value

    return lt.add(zero_padded_lt, scalar_lt, name=scope)


def entry_point_batch(
    input_lts: List[lt.LabeledTensor],
    bp: BatchParameters,
    enqueue_many: bool,
    entry_point_names: List[str],
    name: str = None,
) -> List[lt.LabeledTensor]:
  """Wraps lt.batch with C++ entry points.

  The original and rebatched tensors are given op names derived from
  `entry_point_names`.
  All entry point names begin with 'entry_point'.

  Args:
    input_lts: The input tensors.
    bp: BatchParameters.
    enqueue_many: Batch parameter.
    entry_point_names: The names to give to each of the tensors.
    name: Optional batch op name.

  Returns:
    The rebatched inputs.
  """
  assert len(input_lts) == len(entry_point_names)
  with tf.name_scope(''):
    input_lts = [
        lt.identity(t, name='entry_point_%s_pre_batch' % n)
        for (t, n) in zip(input_lts, entry_point_names)
    ]
  batch_lts = lt.batch(
      input_lts,
      batch_size=bp.size,
      num_threads=bp.num_threads,
      capacity=bp.capacity,
      enqueue_many=enqueue_many,
      name=name)
  with tf.name_scope(''):
    batch_lts = [
        lt.identity(t, name='entry_point_%s_post_batch' % n)
        for (t, n) in zip(batch_lts, entry_point_names)
    ]

  return batch_lts


def softmax_cross_entropy(target_lt: lt.LabeledTensor,
                          mask_lt: lt.LabeledTensor,
                          predicted_lt: lt.LabeledTensor,
                          name: str = None) -> lt.LabeledTensor:
  """Rescaled sparse softmax cross entropy."""
  with tf.name_scope(name, 'softmax_cross_entropy',
                     [target_lt, mask_lt, predicted_lt]) as scope:
    target_lt = lt.transpose(target_lt, ['batch'])
    mask_lt = lt.transpose(mask_lt, ['batch'])
    predicted_lt = lt.transpose(predicted_lt, ['batch', 'class'])

    num_classes = len(predicted_lt.axes['class'])
    target_op = tf.to_int32(tf.round(target_lt.tensor * (num_classes - 1)))

    loss_op = tf.losses.sparse_softmax_cross_entropy(
        logits=predicted_lt, labels=target_op, weights=mask_lt)

    # Scale the cross-entropy loss so that 0.0 remains perfect, and 1.0
    # is the loss incurred by a uniform predictor.
    # Any loss greater than 1.0 would therefore be a really bad sign.
    loss_op /= -1.0 * math.log(1.0 / num_classes)

    return lt.identity(lt.LabeledTensor(loss_op, []), name=scope)


def restore_model(
    restore_directory: str,
    restore_logits: bool,
    restore_global_step: bool = False) -> Callable[[tf.Session], Callable]:
  """Creates a function to restore model parameters."""
  logging.info('Restoring model from %s', restore_directory)
  latest_checkpoint = tf.train.latest_checkpoint(restore_directory)
  logging.info('Restore model checkpoint: %s', latest_checkpoint)
  all_variables = slim.get_model_variables()

  def filter_logits(v):
    if restore_logits:
      return True
    else:
      return 'head' not in v.name

  variables_to_restore = [v for v in all_variables if filter_logits(v)]
  if restore_global_step:
    variables_to_restore.append(tf.train.get_or_create_global_step())

  for v in variables_to_restore:
    logging.info('Variable to restore: %s', (v.name, v.dtype))
  restorer = tf.train.Saver(variables_to_restore)
  return lambda sess: restorer.restore(sess, latest_checkpoint)
