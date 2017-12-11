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
"""Graph-run-time Tensorflow sanity checks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import tensorflow as tf
from typing import Optional

flags = tf.flags
lt = tf.contrib.labeled_tensor

flags.DEFINE_bool('tensorcheck_enable_checks', False,
                  'Whether to enable tensor checks. '
                  'The checks create graph ops and add to runtime overhead.')

FLAGS = flags.FLAGS


def bounds_unlabeled(lower: float,
                     upper: float,
                     tensor: tf.Tensor,
                     name: Optional[str] = None) -> tf.Tensor:
  """Checks the tensor elements fall in the given bounds.

  Args:
    lower: The lower bound.
    upper: The upper bound.
    tensor: The input tensor.
    name: Optional op name.

  Returns:
    The input tensor.
  """
  with tf.name_scope(name, 'check_bounds', [tensor]) as scope:
    if FLAGS.tensorcheck_enable_checks:
      lower_bound_op = tf.assert_non_negative(
          tensor - lower, name='lower_bound')
      upper_bound_op = tf.assert_non_positive(
          tensor - upper, name='upper_bound')
      with tf.control_dependencies([lower_bound_op, upper_bound_op]):
        tensor = tf.identity(tensor, name=scope)

    return tensor


def bounds(lower: float,
           upper: float,
           labeled_tensor: lt.LabeledTensor,
           name: Optional[str] = None) -> lt.LabeledTensor:
  """Checks the tensor elements fall in the given bounds.

  Args:
    lower: The lower bound.
    upper: The upper bound.
    labeled_tensor: The input tensor.
    name: Optional op name.

  Returns:
    The input tensor.
  """
  with tf.name_scope(name, 'lt_check_bounds', [labeled_tensor]) as scope:
    return lt.LabeledTensor(
        bounds_unlabeled(lower, upper, labeled_tensor.tensor, name=scope),
        labeled_tensor.axes)


def shape_unlabeled(tensor: tf.Tensor, name: Optional[str] = None) -> tf.Tensor:
  """Checks the shape of the tensor matches between construction- and run-time.

  Graph-construction-time tensor shapes normally have no bearing on run-time
  shapes; they're just there to help humans with bookkeeping.
  So, things can get weird when the shapes assumed at construction time aren't
  the shapes at runtime.
  This is a run-time assert that the shapes do indeed match.

  Args:
    tensor: The input tensor.
    name: Optional op name.

  Returns:
    The input tensor.
  """
  with tf.name_scope(name, 'check_shape', [tensor]) as scope:
    if FLAGS.tensorcheck_enable_checks:
      s = tensor.shape.as_list()
      # Create a version of the tensor without a statically known shape, to
      # force tf.shape to check the shape.
      tensor_copy = tf.placeholder_with_default(
          tensor, shape=[None] * len(tensor.shape.as_list()))
      assert None not in s, s
      assert_op = tf.assert_equal(
          tf.constant(s, dtype=tf.int64),
          tf.shape(tensor_copy, out_type=tf.int64),
          message=
          'Runtime shape does not match shape at graph construction time.')
      with tf.control_dependencies([assert_op]):
        tensor = tf.identity(tensor, name=scope)

      return tensor
    else:
      return tensor


def shape(labeled_tensor: lt.LabeledTensor,
          name: Optional[str] = None) -> lt.LabeledTensor:
  """Checks the shape of the tensor matches between construction- and run-time.

  Graph-construction-time tensor shapes normally have no bearing on run-time
  shapes; they're just there to help humans with bookkeeping.
  So, things can get weird when the shapes assumed at construction time aren't
  the shapes at runtime.
  This is a run-time assert that the shapes do indeed match.

  Args:
    labeled_tensor: The input tensor.
    name: Optional op name.

  Returns:
    The input tensor.
  """
  with tf.name_scope(name, 'tensorcheck.shape', [labeled_tensor]) as scope:
    return lt.LabeledTensor(
        shape_unlabeled(labeled_tensor.tensor, name=scope), labeled_tensor.axes)


def well_defined():
  """A decorator which checks function argument tensors.

  Checked tensors must have the same shape at graph runtime as they had at graph
  construction time.
  Checked tensors must contain only finite values.

  This calls either tf.verify_tensor_all_finite or lt.verify_tensor_all_finite
  on all input tf.Tensors and lt.LabeledTensors.

  Returns:
    A function to use as a decorator.
  """

  def check(f):
    """Check the inputs."""

    # TODO(ericmc): Should we also check kwds?
    @functools.wraps(f)
    def new_f(*args, **kwds):
      """A helper function."""
      new_args = []
      for a in args:
        float_types = [tf.float16, tf.float32, tf.float64]
        if isinstance(a, tf.Tensor):
          new_a = shape_unlabeled(a)
          if a.dtype in float_types:
            new_a = tf.verify_tensor_all_finite(new_a, msg='')
        elif isinstance(a, lt.LabeledTensor):
          new_a = shape(a)
          if a.tensor.dtype in float_types:
            new_a = lt.verify_tensor_all_finite(new_a, message='')
        else:
          new_a = a
        new_args.append(new_a)

      return f(*new_args, **kwds)

    return new_f

  return check
