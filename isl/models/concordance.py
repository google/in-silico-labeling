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
"""A learned fovea model with extra convolutions to reduce upscale noise.

This model is like fovea_core, except:
1) The defovea operations are done with overlap to minimize the screen door
effect.
2) Foveation and defoveation operations are followed by in-scale
model_util.modules,
to make scale changes more gradual and further minimize the screen door
effect.
3) The network is substantially taller, including a taller top tower.
The hope is that by adding depth, the predicted pixels will be more
concordant, thus the network name.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

# pylint: disable=g-bad-import-order
from isl import tensorcheck
from isl import util
from isl.models import model_util

logging = tf.logging
lt = tf.contrib.labeled_tensor
slim = tf.contrib.slim

# The standard convolution sizes for in-scale Minception model_util.modules.
IN_SCALE_EXPANSION_SIZE = 3
IN_SCALE_REDUCTION_SIZE = 1

# We project to a dimension of this size before learning layers with
# nonlinearities.
INITIAL_PROJECTION_DIMENSION = 16


@tensorcheck.well_defined()
def core(
    base_depth: int,
    is_train: bool,
    input_op: tf.Tensor,
    name: str = None,
) -> tf.Tensor:
  """A learned fovea model with extra convolutions to reduce upscale noise.

  Args:
    base_depth: The depth of a 1x1 layer.
      Used as a multiplier when computing layer depths from size.
    is_train: Whether we're training.
    input_op: The input.
    name: Optional op name.

  Returns:
    The output of the core model as an embedding layer.
    Network heads should take this layer as input.
  """
  with tf.name_scope(name, 'concordance_core', [input_op]) as scope:
    # Ensure the input data is in the range [0.0, 1.0].
    input_op = tensorcheck.bounds_unlabeled(0.0, 1.0, input_op)

    input_op = slim.conv2d(
        input_op, INITIAL_PROJECTION_DIMENSION, [1, 1], activation_fn=None)

    # These are the layer sizes (size == width == height) for the lower part
    # of the network (the part that comes before spatial merging).
    # Each row is a single tower, and each tower operates at a different
    # spatial scale.
    # Scale i corresponds to a rescaling factor of 2 ^ i; so 0 is the original
    # scale, 1 is 2x downscaled, 2 is 4x, etc.
    # There are 5 scales total, and the rescaling operations always down- or
    # upscale by a factor of 2.
    # So, to get to scale 4 takes 4 downscalings.
    lls = [
        [
            72, 70, 68, 66, 64, 62, 60, 58, 56, 54, 52, 50, 48, 46, 44, 42, 40,
            38, 36
        ],
        [
            102, 50, 48, 46, 44, 42, 40, 38, 36, 34, 32, 30, 28, 26, 24, 22, 20,
            38, 36
        ],
        [
            146, 72, 70, 34, 32, 30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 22, 20,
            38, 36
        ],
        [
            202, 100, 98, 48, 46, 22, 20, 18, 16, 14, 12, 10, 8, 14, 12, 22, 20,
            38, 36
        ],
        [
            250, 124, 122, 60, 58, 28, 26, 12, 10, 8, 6, 10, 8, 14, 12, 22, 20,
            38, 36
        ],
    ]

    # These are the layer sizes for the upper part of the network.
    # This part is simply several in-scale minception model_util.modules
    # stacked on top of each other.
    uls = [36, 34, 32, 30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8]

    [_, num_rows, num_columns, _] = input_op.shape.as_list()
    assert num_rows == lls[4][0], input_op.shape.as_list()
    assert num_rows == num_columns

    # The number of in-scale Minception model_util.modules for each scale.
    def num_lower_recursive_stacks(s):
      return 18 - 4 * s

    num_upper_recursive_stacks = len(uls) - 1
    num_scales = len(lls)

    def foveate(final_size: int, op: tf.Tensor, name: str) -> tf.Tensor:
      return model_util.learned_fovea(is_train, base_depth, True, final_size,
                                      op, name)

    scale_ops = []
    for s in range(num_scales):
      if s == 0:
        scale_op = util.crop_center_unlabeled(lls[0][0], input_op)
      elif s == 1:
        scale_op = util.crop_center_unlabeled(lls[1][0], input_op)
        scale_op = foveate(lls[1][2], scale_op, 'downscale_1_0')
      elif s == 2:
        scale_op = util.crop_center_unlabeled(lls[2][0], input_op)
        # Note we're skipping every other layer here and below because
        # `foveate` is composed of two model_util.modules
        # (`add_in_scale_model_util.module` is True).
        scale_op = foveate(lls[2][2], scale_op, 'downscale_2_0')
        scale_op = foveate(lls[2][4], scale_op, 'downscale_2_1')
      elif s == 3:
        scale_op = util.crop_center_unlabeled(lls[3][0], input_op)
        scale_op = foveate(lls[3][2], scale_op, 'downscale_3_0')
        scale_op = foveate(lls[3][4], scale_op, 'downscale_3_1')
        scale_op = foveate(lls[3][6], scale_op, 'downscale_3_2')
      elif s == 4:
        # There's no need to crop.
        scale_op = foveate(lls[4][2], input_op, 'downscale_4_0')
        scale_op = foveate(lls[4][4], scale_op, 'downscale_4_1')
        scale_op = foveate(lls[4][6], scale_op, 'downscale_4_2')
        scale_op = foveate(lls[4][8], scale_op, 'downscale_4_3')
      else:
        raise AssertionError

      logging.info('scale %d tower input shape: %s', s,
                   str(scale_op.shape.as_list()))
      scale_ops.append(scale_op)

    multiscale_tower_ops = []
    for s in range(num_scales):
      recursive_op = scale_ops[s]

      for r in range(num_lower_recursive_stacks(s)):
        final_size = recursive_op.shape.as_list()[1] - 2
        recursive_op = model_util.module(
            IN_SCALE_EXPANSION_SIZE,
            IN_SCALE_REDUCTION_SIZE,
            model_util.size_to_depth(base_depth, final_size, True),
            model_util.size_to_depth(base_depth, final_size, False),
            is_deconv=False,
            add_bias=False,
            min_depth_from_residual=True,
            is_train=is_train,
            input_op=recursive_op,
            name='lower_scale_%d_recursion_%d' % (s, r))

      num_recursive_rows = recursive_op.shape.as_list()[1]
      if s == 0:
        assert num_recursive_rows == lls[0][-1], num_recursive_rows
      elif s == 1:
        assert num_recursive_rows == lls[1][-3], num_recursive_rows
      elif s == 2:
        assert num_recursive_rows == lls[2][-5], num_recursive_rows
      elif s == 3:
        assert num_recursive_rows == lls[3][-7], num_recursive_rows
      elif s == 4:
        assert num_recursive_rows == lls[4][-9], num_recursive_rows
      else:
        raise AssertionError

      multiscale_tower_ops.append(recursive_op)

    def defoveate(final_size: int, op: tf.Tensor, name: str) -> tf.Tensor:
      return model_util.learned_defovea(is_train, base_depth, True, True,
                                        final_size, op, name)

    deconv_ops = []
    for s in range(num_scales):
      recursive_op = multiscale_tower_ops[s]

      if s == 0:
        deconv_op = recursive_op
      elif s == 1:
        deconv_op = defoveate(lls[1][-1], recursive_op, 'upscale_1_0')
      elif s == 2:
        # Note we're skipping every other layer here and below because
        # `defoveate` is composed of two model_util.modules
        # (`add_in_scale_model_util.module` is True).
        deconv_op = defoveate(lls[2][-3], recursive_op, 'upscale_2_0')
        deconv_op = defoveate(lls[2][-1], deconv_op, 'upscale_2_1')
      elif s == 3:
        deconv_op = defoveate(lls[3][-5], recursive_op, 'upscale_3_0')
        deconv_op = defoveate(lls[3][-3], deconv_op, 'upscale_3_1')
        deconv_op = defoveate(lls[3][-1], deconv_op, 'upscale_3_2')
      elif s == 4:
        deconv_op = defoveate(lls[4][-7], recursive_op, 'upscale_4_0')
        deconv_op = defoveate(lls[4][-5], deconv_op, 'upscale_4_1')
        deconv_op = defoveate(lls[4][-3], deconv_op, 'upscale_4_2')
        deconv_op = defoveate(lls[4][-1], deconv_op, 'upscale_4_3')
      else:
        raise AssertionError

      deconv_ops.append(deconv_op)

    recursive_op = tf.concat(deconv_ops, 3)
    assert recursive_op.shape.as_list()[1] == uls[0]
    for r in range(num_upper_recursive_stacks):
      final_size = recursive_op.shape.as_list()[1] - 2
      recursive_op = model_util.module(
          IN_SCALE_EXPANSION_SIZE,
          IN_SCALE_REDUCTION_SIZE,
          model_util.size_to_depth(base_depth, final_size, True),
          model_util.size_to_depth(base_depth, final_size, False),
          is_deconv=False,
          add_bias=False,
          min_depth_from_residual=True,
          is_train=is_train,
          input_op=recursive_op,
          name='upper_recursion_%d' % (r))

    assert recursive_op.shape.as_list()[1] == uls[-1]
    return tf.identity(recursive_op, name=scope)
