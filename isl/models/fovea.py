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
"""A Minception model with learned scale changes and balanced paths.

This model is like Minception, except:
1) Downscales and upscales are always learned, using striding convolutions
or deconvolutions.
2) Layer depths are set so that all Minception model_util.modules require
approximately the same number of floating point operations to compute (so as
layer rows and columns decrease, the depth increases).
3) All paths through the network, from input to output, have the same length.

It appears to be at least as accurate as the original Minception model, and
at the time of writing is still converging.
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

# The standard convolution sizes for in-scale Minception model_util.modules.
IN_SCALE_EXPANSION_SIZE = 3
IN_SCALE_REDUCTION_SIZE = 1


@tensorcheck.well_defined()
def core(base_depth: int, is_train: bool, input_op: tf.Tensor,
         name: str = None) -> tf.Tensor:
  """A Minception model with learned scale changes and balanced paths.

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
  with tf.name_scope(name, 'fovea_core', [input_op]) as scope:
    # Ensure the input data is in the range [0.0, 1.0].
    input_op = tensorcheck.bounds_unlabeled(0.0, 1.0, input_op)

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
        # This tower lives entirely in scale 0, the native scale.
        [36, 34, 32, 30, 28, 26, 24, 22, 20, 18, 16],
        # This tower switches to scale 1 in the first step, and back to scale 0
        # in the last step.
        [50, 24, 22, 20, 18, 16, 14, 12, 10, 8, 16],
        # This tower switches to scale 2 in the first two steps, and back to
        # scale 0 in the last two steps.
        [70, 34, 16, 14, 12, 10, 8, 6, 4, 8, 16],
        # This tower switches to scale 3 in the first three steps, and back
        # to scale 0 in the last three steps.
        [94, 46, 22, 10, 8, 6, 4, 2, 4, 8, 16],
        # This tower switches to scale 4 in the first four steps, and back
        # to scale 0 in the last four steps.
        [110, 54, 26, 12, 5, 3, 1, 2, 4, 8, 16],
    ]

    # These are the layer sizes for the upper part of the network.
    # This part is simply several in-scale minception
    # model_util.modules stacked on top of each other.
    uls = [16, 14, 12, 10, 8]

    [_, num_rows, num_columns, _] = input_op.shape_as_list()
    assert num_rows == lls[4][0]
    assert num_rows == num_columns

    # The number of in-scale Minception model_util.modules for each scale.
    num_lower_recursive_stacks = lambda s: 10 - 2 * s
    num_upper_recursive_stacks = len(uls) - 1
    num_scales = len(lls)

    def foveate(final_size: int, op: tf.Tensor, name: str) -> tf.Tensor:
      return model_util.learned_fovea(is_train, base_depth, False, final_size,
                                      op, name)

    scale_ops = []
    for s in range(num_scales):
      if s == 0:
        scale_op = util.crop_center_unlabeled(lls[0][0], input_op)
      elif s == 1:
        scale_op = util.crop_center_unlabeled(lls[1][0], input_op)
        scale_op = foveate(lls[1][1], scale_op, 'downscale_1_0')
      elif s == 2:
        scale_op = util.crop_center_unlabeled(lls[2][0], input_op)
        scale_op = foveate(lls[2][1], scale_op, 'downscale_2_0')
        scale_op = foveate(lls[2][2], scale_op, 'downscale_2_1')
      elif s == 3:
        scale_op = util.crop_center_unlabeled(lls[3][0], input_op)
        scale_op = foveate(lls[3][1], scale_op, 'downscale_3_0')
        scale_op = foveate(lls[3][2], scale_op, 'downscale_3_1')
        scale_op = foveate(lls[3][3], scale_op, 'downscale_3_2')
      elif s == 4:
        # There's no need to crop.
        scale_op = foveate(lls[4][1], input_op, 'downscale_4_0')
        scale_op = foveate(lls[4][2], scale_op, 'downscale_4_1')
        scale_op = foveate(lls[4][3], scale_op, 'downscale_4_2')
        scale_op = foveate(lls[4][4], scale_op, 'downscale_4_3')
      else:
        raise AssertionError

      logging.info('scale %d tower input shape: %s', s,
                   str(scale_op.shape_as_list()))
      scale_ops.append(scale_op)

    multiscale_tower_ops = []
    for s in range(num_scales):
      recursive_op = scale_ops[s]

      for r in range(num_lower_recursive_stacks(s)):
        final_size = recursive_op.shape_as_list()[1] - 2
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

      num_recursive_rows = recursive_op.shape_as_list()[1]
      if s == 0:
        assert num_recursive_rows == lls[0][-1], num_recursive_rows
      elif s == 1:
        assert num_recursive_rows == lls[1][-2], num_recursive_rows
      elif s == 2:
        assert num_recursive_rows == lls[2][-3], num_recursive_rows
      elif s == 3:
        assert num_recursive_rows == lls[3][-4], num_recursive_rows
      elif s == 4:
        assert num_recursive_rows == lls[4][-5], num_recursive_rows
      else:
        raise AssertionError

      multiscale_tower_ops.append(recursive_op)

    def defoveate(final_size: int, op: tf.Tensor, name: str) -> tf.Tensor:
      return model_util.learned_defovea(is_train, base_depth, False, False,
                                        final_size, op, name)

    deconv_ops = []
    for s in range(num_scales):
      recursive_op = multiscale_tower_ops[s]

      if s == 0:
        deconv_op = recursive_op
      elif s == 1:
        deconv_op = defoveate(lls[1][-1], recursive_op, 'upscale_1_0')
      elif s == 2:
        deconv_op = defoveate(lls[2][-2], recursive_op, 'upscale_2_0')
        deconv_op = defoveate(lls[2][-1], deconv_op, 'upscale_2_1')
      elif s == 3:
        deconv_op = defoveate(lls[3][-3], recursive_op, 'upscale_3_0')
        deconv_op = defoveate(lls[3][-2], deconv_op, 'upscale_3_1')
        deconv_op = defoveate(lls[3][-1], deconv_op, 'upscale_3_2')
      elif s == 4:
        deconv_op = defoveate(lls[4][-4], recursive_op, 'upscale_4_0')
        deconv_op = defoveate(lls[4][-3], deconv_op, 'upscale_4_1')
        deconv_op = defoveate(lls[4][-2], deconv_op, 'upscale_4_2')
        deconv_op = defoveate(lls[4][-1], deconv_op, 'upscale_4_3')
      else:
        raise AssertionError

      deconv_ops.append(deconv_op)
    recursive_op = tf.concat(deconv_ops, 3)
    assert recursive_op.shape_as_list()[1] == uls[0]
    for r in range(num_upper_recursive_stacks):
      final_size = recursive_op.shape_as_list()[1] - 2
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

    assert recursive_op.shape_as_list()[1] == uls[-1]
    return tf.identity(recursive_op, name=scope)
