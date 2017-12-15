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
"""Tools for data augmentation."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from typing import NamedTuple, Tuple

# pylint: disable=g-bad-import-order
from isl import util

lt = tf.contrib.labeled_tensor

# The probability each corruption operation will be applied.
CORRUPTION_PROBABILITY = 0.5


def corrupt(offset_standard_deviation: float,
            multiplier_standard_deviation: float,
            noise_standard_deviation: lt.LabeledTensor,
            labeled_tensor: str = None,
            name=None) -> lt.LabeledTensor:
  """Corrupt the given image tensor with pseudo-random noise.

  Args:
    offset_standard_deviation: Standard deviation of per-image intensity offset.
    multiplier_standard_deviation: Standard deviation of per-image intensity
      multiplier.
      Equivalent to varying exposure time.
    noise_standard_deviation: Standard deviation of per-pixel intensity offset.
      Equivalent to shot noise.
    labeled_tensor: The input tensor.
    name: Optional op name.

  Returns:
    The tensor input * multiplier + offset + noise.
  """
  with tf.name_scope(name, 'corrupt', [labeled_tensor]) as scope:
    labeled_tensor = lt.transpose(labeled_tensor,
                                  ['batch', 'row', 'column', 'channel'])
    [batch_size, num_rows, num_columns,
     num_channels] = labeled_tensor.tensor.shape.as_list()

    axes = ['batch', 'row', 'column', 'channel']
    offset_lt = lt.LabeledTensor(
        tf.random_normal(
            [batch_size, num_channels],
            mean=0.0,
            stddev=offset_standard_deviation,
            seed=0), ['batch', 'channel'])
    multiplier_lt = lt.LabeledTensor(
        tf.random_normal(
            [batch_size, num_channels],
            mean=1.0,
            stddev=multiplier_standard_deviation,
            seed=0), ['batch', 'channel'])
    noise_lt = lt.LabeledTensor(
        tf.random_normal(
            [batch_size, num_rows, num_columns, num_channels],
            mean=0.0,
            stddev=noise_standard_deviation,
            seed=0), axes)

    def do_corrupt() -> tf.Tensor:
      return tf.random_uniform(()) < CORRUPTION_PROBABILITY

    corrupt_lt = lt.LabeledTensor(
        tf.cond(
            tf.logical_and(do_corrupt(), multiplier_standard_deviation > 0),
            lambda: labeled_tensor * multiplier_lt, lambda: labeled_tensor),
        axes=labeled_tensor.axes)

    corrupt_lt = lt.LabeledTensor(
        tf.cond(
            tf.logical_and(do_corrupt(), offset_standard_deviation > 0),
            lambda: corrupt_lt + offset_lt, lambda: corrupt_lt),
        axes=labeled_tensor.axes)

    corrupt_lt = lt.LabeledTensor(
        tf.cond(
            tf.logical_and(do_corrupt(), noise_standard_deviation > 0),
            lambda: corrupt_lt + noise_lt, lambda: corrupt_lt),
        axes=labeled_tensor.axes)

    clip_lt = lt.LabeledTensor(
        tf.clip_by_value(corrupt_lt.tensor, 0.0, 1.0, name=scope),
        labeled_tensor.axes)

    return clip_lt


# pylint: disable=invalid-name
AugmentParameters = NamedTuple('AugmentParameters', [
    ('offset_standard_deviation', float),
    ('multiplier_standard_deviation', float),
    ('noise_standard_deviation', float),
])

# pylint: enable=invalid-name


def _random_flip_and_rotation(image_lt: lt.LabeledTensor) -> lt.LabeledTensor:
  image_lt = lt.transpose(image_lt, ['batch', 'row', 'column', 'channel'])

  flip_op = tf.map_fn(tf.image.random_flip_left_right, image_lt.tensor)

  num_rotations_op = tf.random_uniform((), 0, 4, dtype=tf.int32)
  rotate_op = tf.map_fn(lambda image: tf.image.rot90(image, k=num_rotations_op),
                        flip_op)

  return lt.LabeledTensor(rotate_op, axes=image_lt.axes)


def augment(ap: AugmentParameters,
            input_lt: lt.LabeledTensor,
            target_lt: lt.LabeledTensor,
            name: str = None) -> Tuple[lt.LabeledTensor, lt.LabeledTensor]:
  """Apply data augmentation to the given input and target tensors.

  Args:
    ap:  AugmentParameters.
    input_lt: An input tensor with canonical axes.
    target_lt: A target tensor with canonical axes.
    name: Optional op name.

  Returns:
    The augmented input and target tensors.
    Both tensors are rotated and flipped, and the input tensor additionally
    has added noise.
  """
  with tf.name_scope(name, 'augment', [input_lt, target_lt]) as scope:
    input_lt = lt.transpose(input_lt, util.CANONICAL_AXIS_ORDER)
    target_lt = lt.transpose(target_lt, util.CANONICAL_AXIS_ORDER)

    input_rc = lt.ReshapeCoder(['z', 'channel', 'mask'], ['channel'])
    input_reshape_lt = input_rc.encode(input_lt)
    target_rc = lt.ReshapeCoder(['z', 'channel', 'mask'], ['channel'])
    target_reshape_lt = target_rc.encode(target_lt)

    merge_lt = lt.concat([input_reshape_lt, target_reshape_lt], 'channel')
    flip_lt = _random_flip_and_rotation(merge_lt)

    num_reshaped_input_channels = len(input_reshape_lt.axes['channel'])
    input_lt = input_rc.decode(flip_lt[:, :, :, :num_reshaped_input_channels])
    target_lt = target_rc.decode(flip_lt[:, :, :, num_reshaped_input_channels:])

    # Select out the input signal channel and add noise to it.
    input_pixels_lt = lt.select(input_lt, {'mask': util.slice_1(False)})
    rc = lt.ReshapeCoder(['z', 'channel', 'mask'], ['channel'])
    input_pixels_lt = rc.decode(
        corrupt(ap.offset_standard_deviation, ap.multiplier_standard_deviation,
                ap.noise_standard_deviation, rc.encode(input_pixels_lt)))
    input_lt = lt.concat(
        [input_pixels_lt,
         lt.select(input_lt, {
             'mask': util.slice_1(True)
         })],
        'mask',
        name=scope + 'input')

    target_lt = lt.identity(target_lt, name=scope + 'target')

    return input_lt, target_lt
