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
"""Tools for building TensorFlow networks."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf
from typing import List, Tuple

# pylint: disable=g-bad-import-order
from isl import tensorcheck
from isl import util

logging = tf.logging
lt = tf.contrib.labeled_tensor
slim = tf.contrib.slim

# The standard convolution sizes for in-scale Minception modules.
IN_SCALE_EXPANSION_SIZE = 3
IN_SCALE_REDUCTION_SIZE = 1


@tensorcheck.well_defined()
def residual_v2_conv(
    kernel_size: int,
    stride: int,
    depth: int,
    is_deconv: bool,
    add_max_pool: bool,
    add_bias: bool,
    is_train: bool,
    input_op: tf.Tensor,
    name: str = None,
) -> tf.Tensor:
  """Creates a residual convolution in the style of He et al. April 2016.

  This is the second version of their proposed residual structure, where the
  order of operations is batch_norm -> activation -> convolution.
  We use RELU and TANH activations, and we optionally add a max pool.

  Args:
    kernel_size: The size of the kernel.
    stride: The stride of the convolution.
    depth: The depth of the reduction layer.
    is_deconv: Whether this is a deconvolution.
    add_max_pool: Whether to add a parallel max pool with the same parameters
      as the convolution.
    add_bias: Whether to add bias to the convolution.
    is_train: Whether we're training this graph.
    input_op: The input.
    name: An optional op name.

  Returns:
    The tensor output of residual convolution.
  """
  with tf.variable_scope(name, 'residual_v2_conv', [input_op]) as scope:
    [_, num_rows, num_columns, _] = input_op.shape.as_list()
    if not is_deconv:
      assert num_rows >= kernel_size
      assert num_columns >= kernel_size
      # Make sure we can do a valid convolution.
      assert (num_rows - kernel_size) % stride == 0
      assert (num_columns - kernel_size) % stride == 0

    # In the future it may be necessary to set epsilon to a larger value
    # than the default here.
    bn_op = slim.batch_norm(input_op, is_training=is_train, scale=True)
    concat_op = tf.concat([tf.nn.relu(bn_op), tf.nn.tanh(bn_op)], 3)

    with slim.arg_scope(
        [slim.conv2d, slim.conv2d_transpose, slim.max_pool2d],
        kernel_size=kernel_size,
        stride=stride,
        padding='VALID'):
      if add_bias:
        biases_initializer = tf.zeros_initializer()
      else:
        biases_initializer = None
      with slim.arg_scope(
          [slim.conv2d, slim.conv2d_transpose],
          inputs=concat_op,
          num_outputs=depth,
          activation_fn=None,
          biases_initializer=biases_initializer):
        if is_deconv:
          conv_op = slim.conv2d_transpose()
        else:
          conv_op = slim.conv2d()

      if add_max_pool:
        assert not is_deconv
        assert kernel_size > 1
        return tf.concat(
            [conv_op, slim.max_pool2d(input_op)], 3, name=scope.name)
      else:
        return tf.identity(conv_op, name=scope.name)


@tensorcheck.well_defined()
def add_with_mismatched_depths(
    a_op: tf.Tensor,
    b_op: tf.Tensor,
    pad: bool,
    name: str = None,
) -> tf.Tensor:
  """Add two tensors together, padding or cropping in depth if necessary.

  Args:
    a_op: A tensor to add.
    b_op: A tensor to add.
    pad: Whether to pad the smaller depth with zeros to make addition work.
      Else, the larger depth is cropped.
    name: Optional op name.

  Returns:
    The sum of the two tensors, possibly after zero-padding or cropping.
  """
  with tf.variable_scope(name, 'add_with_mismatched_depths',
                         [a_op, b_op]) as scope:
    assert len(a_op.shape.as_list()) == 4
    assert len(b_op.shape.as_list()) == 4
    assert a_op.shape.as_list()[:3] == b_op.shape.as_list()[:3]

    a_depth = a_op.shape.as_list()[3]
    b_depth = b_op.shape.as_list()[3]

    if a_depth != b_depth:
      if pad:
        pad_size = abs(a_depth - b_depth)
        zeros_op = tf.zeros(a_op.shape.as_list()[:3] + [pad_size])
        if a_depth < b_depth:
          a_op = tf.concat([a_op, zeros_op], 3)
        else:
          b_op = tf.concat([b_op, zeros_op], 3)
      else:
        min_depth = min(a_depth, b_depth)
        if a_depth > min_depth:
          a_op = a_op[:, :, :, :min_depth]
        else:
          b_op = b_op[:, :, :, :min_depth]

    return tf.add(a_op, b_op, name=scope.name)


@tensorcheck.well_defined()
def passthrough(
    kernel_size: int,
    stride: int,
    is_deconv: bool,
    input_op: tf.Tensor,
    name: str = None,
) -> tf.Tensor:
  """Get the residual identity for the given convolution or pooling shape.

  We assume the convolution or pooling padding is VALID and the size such that
  all pixels are inputs to the next layer.

  Args:
    kernel_size: The size of the convolution or pooling kernel.
    stride: The convolution or pooling stride.
    is_deconv: Whether this is for a deconvolution.
    input_op: The input tensor.
    name: Optional op name.

  Returns:
    The passthrough tensor to use as the residual connection.
    In the case the stride is 1, it is simply a center crop of the input.
    Generally, it is an average pooling of a center crop.
  """
  with tf.name_scope(name, 'passthrough', [input_op]) as scope:
    # TODO(ericmc): Support more configurations.
    conv_valid_3 = not is_deconv and (kernel_size == 3 and stride == 1)
    conv_valid_4 = not is_deconv and (kernel_size == 4 and stride == 2)
    deconv_valid_2 = is_deconv and (kernel_size == 2 and stride == 2)
    deconv_valid_4 = is_deconv and (kernel_size == 4 and stride == 2)
    assert conv_valid_3 or conv_valid_4 or deconv_valid_2 or deconv_valid_4

    if kernel_size == 3:
      pool_size = 1
    else:
      pool_size = 2

    [_, num_rows, num_columns, _] = input_op.shape.as_list()
    assert num_rows == num_columns
    if is_deconv:
      interior_op = tf.tile(input_op, [1, 1, 1, 4])
      interior_op = tf.depth_to_space(interior_op, 2)
      if kernel_size == 2:
        return tf.identity(interior_op, name=scope)
      else:
        return tf.pad(
            interior_op, [[0, 0], [1, 1], [1, 1], [0, 0]],
            mode='SYMMETRIC',
            name=scope)
    else:
      interior_op = util.crop_center_unlabeled(num_rows - 2, input_op)

      # Do simple scaling via averaging.
      return tf.nn.avg_pool(
          interior_op, [1, pool_size, pool_size, 1], [1, stride, stride, 1],
          padding='VALID',
          name=scope)


@tensorcheck.well_defined()
def module(
    expansion_kernel_size: int,
    expansion_stride: int,
    expansion_depth: int,
    reduction_depth: int,
    is_deconv: bool,
    add_bias: bool,
    min_depth_from_residual: bool,
    is_train: bool,
    input_op: tf.Tensor,
    name: tf.Tensor = None,
) -> tf.Tensor:
  """Creates a Minception module with residual v2 connections.

  Args:
    expansion_kernel_size: The size of the expansion kernel.
    expansion_stride: Stride for the expansion convolution and max pool.
    expansion_depth: The depth of the expansion layer.
    reduction_depth: The depth of the reduction layer.
    is_deconv: Whether this is a deconvolution.
    add_bias: Whether to add a bias to the final residual v2
      convolution, if we use v2.
    min_depth_from_residual: Whether to ensure the convolution depths are at
      least the residual depths.
    is_train: Whether we're training this graph.
    input_op: The input.
    name: An optional op name.

  Returns:
    The tensor output of the Minception module.
  """
  with tf.variable_scope(name, 'module', [input_op]) as scope:
    if min_depth_from_residual:
      residual_depth = input_op.shape.as_list()[3]
      reduction_depth = max(reduction_depth, residual_depth)
      # TODO(ericmc): Adjusting the expansion depth here appears to help the
      # model, but it feels hacky and I suspect it's hiding an imbalance in the
      # way depth is apportioned by layer size.
      expansion_depth = max(expansion_depth, residual_depth)

    expand_op = residual_v2_conv(
        expansion_kernel_size,
        expansion_stride,
        expansion_depth,
        is_deconv=is_deconv,
        add_max_pool=not is_deconv,
        add_bias=False,
        is_train=is_train,
        input_op=input_op,
        name='expand_rv2')
    reduce_op = residual_v2_conv(
        1,
        1,
        reduction_depth,
        is_deconv=False,
        add_max_pool=False,
        add_bias=add_bias,
        is_train=is_train,
        input_op=expand_op,
        name='reduce_rv2')

    input_op = passthrough(
        expansion_kernel_size,
        expansion_stride,
        is_deconv=is_deconv,
        input_op=input_op)
    activation_op = add_with_mismatched_depths(
        input_op, reduce_op, pad=not is_deconv, name=scope)

    tf.summary.histogram('activation', activation_op)
    return activation_op


def next_even(n):
  n = int(math.ceil(n))
  return n + (n % 2)


def size_to_depth(base_depth: int, size: int, is_expand: bool) -> int:
  """The depth of a layer given its number or rows (assuming square)."""
  reduction_depth = math.ceil(base_depth / float(size))
  if is_expand:
    # This magic number is an artifact of an old hyperparameter search.
    return next_even((211.0 / 39.0) * reduction_depth)
  else:
    return next_even(reduction_depth)


def learned_fovea(
    is_train: bool,
    base_depth: int,
    add_in_scale_module: bool,
    final_size: int,
    op: tf.Tensor,
    name: str,
) -> tf.Tensor:
  """Downscale the layer by a factor of two.

  Args:
    is_train: Whether we're training.
    base_depth: Used to calculate the depths of the layers.
    add_in_scale_module: Whether to follow the foveation with a single
      in-scale Minception module.
    final_size: The expected size of the output.
    op: The input op.
    name: Op name.

  Returns:
    A downscaled layer.
  """
  op = module(
      4,
      2,
      size_to_depth(base_depth, final_size, True),
      size_to_depth(base_depth, final_size, False),
      is_deconv=False,
      add_bias=False,
      min_depth_from_residual=True,
      is_train=is_train,
      input_op=op,
      name=name)
  if add_in_scale_module:
    op = module(
        IN_SCALE_EXPANSION_SIZE,
        IN_SCALE_REDUCTION_SIZE,
        size_to_depth(base_depth, final_size, True),
        size_to_depth(base_depth, final_size, False),
        is_deconv=False,
        add_bias=False,
        min_depth_from_residual=True,
        is_train=is_train,
        input_op=op,
        name=name + '_in_scale')
  assert op.shape.as_list()[1] == final_size, op.shape.as_list()
  assert op.shape.as_list()[2] == final_size, op.shape.as_list()
  return op


def learned_defovea(
    is_train: bool,
    base_depth: int,
    use_overlap: bool,
    add_in_scale_module: bool,
    final_size: int,
    op: tf.Tensor,
    name: str,
) -> tf.Tensor:
  """Upscale the layer by a factor of two.

  Args:
    is_train: Whether we're training.
    base_depth: Used to calculate the depths of the layers.
    use_overlap: Whether to use an overlapping conv2d_transpose kernel.
      This is intended to reduce screen door artifacts.
    add_in_scale_module: Whether to follow the foveation with a single
      in-scale Minception module.
      This is intended to reduce screen door artifacts.
    final_size: The expected size of the output.
    op: The input op.
    name: Op name.

  Returns:
    An upscaled layer.
  """
  if use_overlap:
    # Use overlap then crop to avoid edge artifacts.
    op = module(
        4,
        2,
        size_to_depth(base_depth - 4, final_size, True),
        size_to_depth(base_depth - 4, final_size, False),
        is_deconv=True,
        add_bias=False,
        # We need to shed depth as we increase size.
        min_depth_from_residual=False,
        is_train=is_train,
        input_op=op,
        name=name)
    op = util.crop_center_unlabeled(op.shape.as_list()[1] - 4, op)
  else:
    # Deconvolve without overlap to avoid edge artifacts.
    # This may introduce "screen door" artifacts.
    op = module(
        2,
        2,
        size_to_depth(base_depth, final_size, True),
        size_to_depth(base_depth, final_size, False),
        is_deconv=True,
        add_bias=False,
        # We need to shed depth as we increase size.
        min_depth_from_residual=False,
        is_train=is_train,
        input_op=op,
        name=name)
  if add_in_scale_module:
    op = module(
        IN_SCALE_EXPANSION_SIZE,
        IN_SCALE_REDUCTION_SIZE,
        size_to_depth(base_depth, final_size, True),
        size_to_depth(base_depth, final_size, False),
        is_deconv=False,
        add_bias=False,
        min_depth_from_residual=True,
        is_train=is_train,
        input_op=op,
        name=name + '_in_scale')
  assert op.shape.as_list()[1] == final_size, (op.shape.as_list(), final_size)
  assert op.shape.as_list()[2] == final_size, op.shape.as_list()
  return op


@tensorcheck.well_defined()
def add_head(
    num_classes: int,
    is_train: bool,
    is_residual_conv: bool,
    input_op: tf.Tensor,
    name: str = None,
) -> tf.Tensor:
  """Add a head to the network.

  Args:
    num_classes: Number of classes in the prediction.
    is_train: Whether we're training this network.
    is_residual_conv: If True, head will use residual_v2_conv. If False, head
      will use slim.conv2d.
    input_op: The input embedding layer.
    name: Optional op name.

  Returns:
    A tensor with shape [batch, row, column, class].
  """
  with tf.variable_scope(name, 'head', [input_op]) as scope:
    [batch_size, num_rows, num_columns, _] = input_op.shape.as_list()

    if is_residual_conv:
      logit_op = residual_v2_conv(
          1,
          stride=1,
          depth=num_classes,
          is_deconv=False,
          add_max_pool=False,
          add_bias=True,
          is_train=is_train,
          input_op=input_op)
    else:
      logit_op = slim.conv2d(
          input_op,
          num_classes, [1, 1],
          activation_fn=None,
          biases_initializer=tf.zeros_initializer())

    tf.summary.histogram('logit', logit_op)

    return tf.reshape(
        logit_op, [batch_size, num_rows, num_columns, num_classes],
        name=scope.name)


class PredictionParameters(object):
  """Convenience class for prediction parameters."""

  def __init__(
      self,
      target_axes: List[Tuple[object, object]],
      num_classes: int,
  ):
    """Construct a PredictionParameters.

    Args:
      target_axes: A list of tuples, where the first element in each tuple is
        the target z-axis values and the second element is the target
        channel-axis values.
      num_classes: The number of pixel classes to predict.
    """
    self.target_axes = [
        lt.Axes([('z', z), ('channel', channel)]) for z, channel in target_axes
    ]
    self.num_classes = num_classes
