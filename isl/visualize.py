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
"""Ops for visualizing tensors."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf
from typing import Tuple

# pylint: disable=g-bad-import-order
from isl import ops
from isl import util

lt = tf.contrib.labeled_tensor

# Some useful colors as RGB intensities.
PURPLE = (102.0 / 255.0, 0.0 / 255.0, 255.0 / 255.0)
RED = (1.0, 0.0, 0.0)
BLUE = (0.0, 0.0, 1.0)
WHITE = (1.0, 1.0, 1.0)
BLACK = (0.0, 0.0, 0.0)
MANGO = (255.0 / 255.0, 130.0 / 255.0, 67.0 / 255.0)
TURQUOISE = (64.0 / 255.0, 224.0 / 255.0, 208.0 / 255.0)
PREDICTION_ORANGE = (1.0, 0.5, 0.0)
TARGET_BLUE = (0.0, 0.5, 1.0)

# Standard image border size when assembling grids of images.
PAD_WIDTH = 4


def to_softmax(
    logit_lt: lt.LabeledTensor,
    name: str = None,
) -> lt.LabeledTensor:
  """Converts a logit tensor to a probability distribution tensor.

  Args:
    logit_lt: The input tensor with canonical prediction axes and logit values.
    name: Optional op name.

  Returns:
    A tensor with probability distributions created via the softmax function.
  """
  with tf.name_scope(name, 'to_softmax', [logit_lt]) as scope:
    logit_lt = lt.transpose(logit_lt, util.CANONICAL_PREDICTION_AXIS_ORDER)

    rc = lt.ReshapeCoder(list(logit_lt.axes.keys())[:-1], ['batch'])
    softmax_op = tf.nn.softmax(rc.encode(logit_lt).tensor)

    softmax_lt = lt.LabeledTensor(
        softmax_op, ['batch', list(logit_lt.axes.values())[-1]])

    return lt.identity(rc.decode(softmax_lt), name=scope)


def add_border(
    color: Tuple[float, float, float],
    size: int,
    labeled_tensor: lt.LabeledTensor,
    name: str = None,
) -> lt.LabeledTensor:
  """Adds a colored border to an image.

  Args:
    color: The color of the border to add.
    size: The size of the border padding.
    labeled_tensor: The input tensor, which must have 'row' and 'column' axes.
    name: Optional op name.

  Returns:
    A tensor with padded 'row' and 'column' dimensions.
    If the input lacked a 'color' axis, this adds the axis ('color',
      ['red', 'green', 'blue']).
  """
  with tf.name_scope(name, 'add_border', [labeled_tensor]) as scope:
    if 'color' not in labeled_tensor.axes:
      split_lts = [labeled_tensor, labeled_tensor, labeled_tensor]
      final_axis_order = list(labeled_tensor.axes.keys()) + ['color']
    else:
      assert (labeled_tensor.axes['color'].labels == (
          'red', 'green', 'blue')), labeled_tensor.axes['color'].labels

      # Make 'color' the first axis.
      transpose_lt = lt.transpose(
          labeled_tensor,
          ['color'] + list(labeled_tensor.axes.remove('color').keys()))
      split_lts = lt.unpack(transpose_lt)

      final_axis_order = list(labeled_tensor.axes.keys())

    assert len(split_lts) == 3

    padded_lts = []
    for (c, split_lt) in zip(color, split_lts):
      padded_lts.append(
          util.pad_constant(split_lt,
                            {'row': (size, size),
                             'column': (size, size)}, c))

    transpose_lt = lt.pack(padded_lts, ('color', ['red', 'green', 'blue']))

    return lt.transpose(transpose_lt, final_axis_order, name=scope)


def summarize_image(image_lt: lt.LabeledTensor, name: str = None) -> tf.Tensor:
  """Registers an image summary for this image.

  Downscales images which use a very large number of pixels.

  Args:
    image_lt: The input image tensor, which must have axes [batch, row, column]
      or [batch, row, column, color].
    name: Optional op name.

  Returns:
    The image summary.
  """
  with tf.name_scope(name, 'summarize_image', [image_lt]) as scope:
    # Downscale an images with more than this many pixels.
    max_num_pixels = 1 << 22

    axes = list(image_lt.axes.keys())
    assert axes[:3] == ['batch', 'row', 'column']
    if len(axes) > 3:
      assert len(axes) == 4
      assert axes[3] == 'color'
    else:
      image_lt = lt.expand_dims(
          image_lt,
          ['batch', 'row', 'column', [('color', ['red', 'green', 'blue'])]])

    num_rows = len(image_lt.axes['row'])
    num_columns = len(image_lt.axes['column'])
    num_pixels = num_rows * num_columns

    if num_pixels > max_num_pixels:
      scale_factor = (float(max_num_pixels) / num_pixels)**0.5

      resized_num_rows = int(scale_factor * num_rows)
      resized_num_columns = int(scale_factor * num_columns)

      resize_op = tf.image.resize_bilinear(
          image_lt.tensor, [resized_num_rows, resized_num_columns])
      image_lt = lt.LabeledTensor(
          resize_op,
          ['batch', 'row', 'column', ('color', ['red', 'green', 'blue'])])

    return tf.summary.image(tensor=image_lt.tensor, name=scope)


def colorize(
    color_scheme: Tuple[float, float, float],
    image_lt: lt.LabeledTensor,
    name: str = None,
) -> lt.LabeledTensor:
  """Colors an uncolored image.

  Args:
    color_scheme: Grayscale intensity values will become intensity values of
      this color.
    image_lt: The input uncolored image.
    name: Optional op name.

  Returns:
    A color image tensor with a new 'color' axis.
  """
  with tf.name_scope(name, 'colorize', [image_lt]) as scope:
    scaled_lts = []
    for c in color_scheme:
      assert c >= 0.0
      assert c <= 1.0
      scaled_lts.append(image_lt * c)

    transpose_lt = lt.pack(scaled_lts, ('color', ['red', 'green', 'blue']))

    return lt.transpose(
        transpose_lt, list(image_lt.axes.keys()) + ['color'], name=scope)


def additive_error(
    target_lt: lt.LabeledTensor,
    predicted_lt: lt.LabeledTensor,
    name: str = None,
) -> lt.LabeledTensor:
  """Displays error using the additive scheme.

  Args:
    target_lt: The ground truth values.
    predicted_lt: The predicted values.
    name: Optional op name.

  Returns:
    The additive visualization of the error.
    Any color on the gray line between black and white is correct, anything
    with a blue tint is a false negative, and anything with an orange tint
    is a false positive.
  """
  with tf.name_scope(name, 'additive_error',
                     [target_lt, predicted_lt]) as scope:
    target_lt = lt.transpose(target_lt, ['batch', 'row', 'column', 'channel'])
    predicted_lt = lt.transpose(predicted_lt,
                                ['batch', 'row', 'column', 'channel'])

    target_lt = colorize(TARGET_BLUE, target_lt)
    predicted_lt = colorize(PREDICTION_ORANGE, predicted_lt)

    return lt.add(target_lt, predicted_lt, name=scope)


def subtractive_error(
    target_lt: lt.LabeledTensor,
    predicted_lt: lt.LabeledTensor,
    name: str = None,
) -> lt.LabeledTensor:
  """Displays error using the difference scheme.

  Args:
    target_lt: The ground truth values.
    predicted_lt: The predicted values.
    name: Optional op name.

  Returns:
    The subtractive visualization of the error.
    Black is correct, anything blue is a false negative, and anything orange
    is a false positive.
  """
  with tf.name_scope(name, 'subtractive_error',
                     [target_lt, predicted_lt]) as scope:
    target_lt = lt.transpose(target_lt, ['batch', 'row', 'column', 'channel'])
    predicted_lt = lt.transpose(predicted_lt,
                                ['batch', 'row', 'column', 'channel'])

    difference_lt = predicted_lt - target_lt

    false_positive_lt = lt.LabeledTensor(
        tf.to_float(difference_lt.tensor > 0), difference_lt.axes)
    false_negative_lt = lt.LabeledTensor(
        tf.to_float(difference_lt.tensor < 0), difference_lt.axes)

    false_positive_lt = difference_lt * false_positive_lt
    false_negative_lt = difference_lt * false_negative_lt * (-1.0)

    false_positive_lt = colorize(PREDICTION_ORANGE, false_positive_lt)
    false_negative_lt = colorize(TARGET_BLUE, false_negative_lt)

    return lt.add(false_positive_lt, false_negative_lt, name=scope)


def cross_entropy_error(
    target_lt: lt.LabeledTensor,
    predicted_lt: lt.LabeledTensor,
    name: str = None,
) -> lt.LabeledTensor:
  """Displays the cross entropy error.

  Args:
    target_lt: The ground truth values as a probability distribution.
    predicted_lt: The predicted values as a probability distribution.
    name: Optional op name.

  Returns:
    The cross entropy visualization of the error, where black is no error
    and white is the error of a uniform predictor.
  """
  with tf.name_scope(name, 'cross_entropy_error',
                     [target_lt, predicted_lt]) as scope:
    epsilon = 0.000001
    product_lt = lt.LabeledTensor(
        tf.log(predicted_lt.tensor + epsilon), predicted_lt.axes) * target_lt
    cross_entropy_lt = lt.reduce_sum(product_lt, ['class']) * (-1.0)

    num_classes = len(predicted_lt.axes['class'])
    uniform_cross_entropy = math.log(num_classes)

    # Scale and clip to focus on small errors and treat all large errors as 1.0.
    cross_entropy_lt /= uniform_cross_entropy
    return lt.LabeledTensor(
        tf.clip_by_value(cross_entropy_lt.tensor, 0.0, 1.0, name=scope),
        cross_entropy_lt.axes)


def canonical_image(
    canonical_lt: lt.LabeledTensor,
    name: str = None,
) -> lt.LabeledTensor:
  """Creates an image displaying the canonical data.

  Args:
    canonical_lt: The input tensor in canonical order.
    name: Optional op name.

  Returns:
    An image.
  """
  with tf.name_scope(name, 'canonical_image', [canonical_lt]) as scope:
    canonical_lt = lt.transpose(canonical_lt, util.CANONICAL_AXIS_ORDER)

    rows = []

    def get_row(
        labeled_tensor: lt.LabeledTensor,
        color: Tuple[float, float, float],
    ):
      labeled_tensor = add_border(color, PAD_WIDTH, labeled_tensor)
      labeled_tensor = lt.transpose(
          labeled_tensor, ['batch', 'row', 'z', 'channel', 'column', 'color'])
      rows.append(
          lt.reshape(labeled_tensor, ['z', 'channel', 'column'], ['column']))

    get_row(lt.select(canonical_lt, {'mask': False}), TURQUOISE)
    get_row(lt.select(canonical_lt, {'mask': True}), MANGO)

    image_lt = lt.concat(rows, 'row', name=scope)

    return image_lt


def error_panel(
    target_lt: lt.LabeledTensor,
    predicted_lt: lt.LabeledTensor,
    name: str = None,
) -> lt.LabeledTensor:
  """Creates a big informative error panel image.

  Args:
    target_lt: The ground truth values in canonical order.
    predicted_lt: The predicted values in canonical prediction order as a
     probability distribution.
    name: Optional op name.

  Returns:
    The error panel.
  """
  with tf.name_scope(name, 'error_panel', [target_lt, predicted_lt]) as scope:
    target_lt = lt.transpose(target_lt, util.CANONICAL_AXIS_ORDER)
    predicted_lt = lt.transpose(predicted_lt,
                                util.CANONICAL_PREDICTION_AXIS_ORDER)

    assert list(target_lt.axes.items())[:-1] == list(
        predicted_lt.axes.items())[:-1], (target_lt.axes, predicted_lt.axes)

    rc = lt.ReshapeCoder(list(predicted_lt.axes.keys())[:-1], ['batch'])
    statistic_lt = rc.decode(
        ops.distribution_statistics(rc.encode(predicted_lt)))

    columns = []

    def get_column(
        labeled_tensor: lt.LabeledTensor,
        color: Tuple[float, float, float],
    ):
      labeled_tensor = add_border(color, PAD_WIDTH, labeled_tensor)
      labeled_tensor = lt.transpose(
          labeled_tensor, ['batch', 'z', 'channel', 'row', 'column', 'color'])
      columns.append(
          lt.reshape(labeled_tensor, ['z', 'channel', 'row'], ['row']))

    # We only show these statistics.
    statistics = ['mode', 'median', 'mean', 'standard_deviation', 'entropy']

    # Show the statistics on the predictions.
    for s in statistics:
      get_column(lt.select(statistic_lt, {'statistic': s}), PURPLE)

    # Show the ground truth target image.
    get_column(lt.select(target_lt, {'mask': False}), TURQUOISE)

    # Show the cross entropy error.
    num_classes = len(predicted_lt.axes['class'])
    cross_entropy_lt = cross_entropy_error(
        util.onehot(num_classes, lt.select(target_lt, {
            'mask': False
        })), predicted_lt)
    get_column(cross_entropy_lt, RED)

    # Show the additive error visualizations.
    for s in statistics[:3]:
      rc = lt.ReshapeCoder(['z', 'channel'], ['channel'])
      error_lt = rc.decode(
          additive_error(
              rc.encode(lt.select(target_lt, {
                  'mask': False
              })), rc.encode(lt.select(statistic_lt, {
                  'statistic': s
              }))))
      get_column(error_lt, WHITE)

    # Show the subtractive error visualizations.
    for s in statistics[:3]:
      rc = lt.ReshapeCoder(['z', 'channel'], ['channel'])
      error_lt = rc.decode(
          subtractive_error(
              rc.encode(lt.select(target_lt, {
                  'mask': False
              })), rc.encode(lt.select(statistic_lt, {
                  'statistic': s
              }))))
      get_column(error_lt, BLACK)

    # Show the pixel presence / absence masks.
    get_column(lt.select(target_lt, {'mask': True}), MANGO)

    panel_lt = lt.concat(columns, 'column', name=scope)

    return panel_lt


def error_panel_from_statistics(
    target_lt: lt.LabeledTensor,
    statistic_lt: lt.LabeledTensor,
    simplify: bool,
    name: str = None,
) -> lt.LabeledTensor:
  """Creates an error panel from statistics using minimal RAM.

  Args:
    target_lt: The ground truth values in canonical order.
    statistic_lt: The canonical statistics of the predicted values.
    simplify: Whether to simplify the error panel.
    name: Optional op name.

  Returns:
    The error panel.
  """
  with tf.name_scope(name, 'error_panel_from_statistics',
                     [target_lt, statistic_lt]) as scope:
    target_lt = lt.transpose(target_lt, util.CANONICAL_AXIS_ORDER)
    statistic_lt = lt.transpose(statistic_lt,
                                util.CANONICAL_STATISTIC_AXIS_ORDER)

    columns = []

    def get_column(
        labeled_tensor: lt.LabeledTensor,
        color: Tuple[float, float, float],
    ):
      labeled_tensor = add_border(color, PAD_WIDTH, labeled_tensor)
      labeled_tensor = lt.transpose(
          labeled_tensor, ['batch', 'z', 'channel', 'row', 'column', 'color'])
      columns.append(
          lt.reshape(labeled_tensor, ['z', 'channel', 'row'], ['row']))

    # We only show these statistics.
    if simplify:
      statistics = ['median']
    else:
      statistics = ['mode', 'median', 'mean', 'standard_deviation']

    # Show the statistics on the predictions.
    for s in statistics:
      get_column(lt.select(statistic_lt, {'statistic': s}), PURPLE)

    # Show the ground truth target image.
    get_column(lt.select(target_lt, {'mask': False}), TURQUOISE)

    if not simplify:
      # Show the additive error visualizations.
      for s in statistics[:3]:
        rc = lt.ReshapeCoder(['z', 'channel'], ['channel'])
        error_lt = rc.decode(
            additive_error(
                rc.encode(lt.select(target_lt, {
                    'mask': False
                })), rc.encode(lt.select(statistic_lt, {
                    'statistic': s
                }))))
        get_column(error_lt, WHITE)

    # Show the subtractive error visualizations.
    for s in statistics[:3]:
      rc = lt.ReshapeCoder(['z', 'channel'], ['channel'])
      error_lt = rc.decode(
          subtractive_error(
              rc.encode(lt.select(target_lt, {
                  'mask': False
              })), rc.encode(lt.select(statistic_lt, {
                  'statistic': s
              }))))
      get_column(error_lt, BLACK)

    if not simplify:
      # Show the pixel presence / absence masks.
      get_column(lt.select(target_lt, {'mask': True}), MANGO)

    panel_lt = lt.concat(columns, 'column', name=scope)

    return panel_lt
