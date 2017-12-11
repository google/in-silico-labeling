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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import tensorflow as tf

# pylint: disable=g-bad-import-order
from isl import controller
from isl import test_util
from isl.models import fovea
from isl.models import model_util

flags = tf.flags
lt = tf.contrib.labeled_tensor
test = tf.test

FLAGS = flags.FLAGS


class PassthroughTest(test_util.Base):

  def setUp(self):
    super(PassthroughTest, self).setUp()

    original_op = tf.to_float(tf.range(0, 2 * 7 * 7 * 5))
    self.original_op = tf.reshape(original_op, [2, 7, 7, 5])

    self.deconv_op = model_util.passthrough(4, 2, True, self.original_op)
    self.conv_op = model_util.passthrough(4, 2, False, self.deconv_op)

  def test_name(self):
    self.assertRegexpMatches(self.deconv_op.name, 'passthrough')
    self.assertRegexpMatches(self.conv_op.name, 'passthrough')

  def test_size(self):
    self.assertListEqual(self.deconv_op.shape_as_list(), [2, 16, 16, 5])

  def test_bijection(self):
    self.assert_tensors_equal(self.original_op, self.conv_op)

  def test_padding(self):
    original_op = tf.ones([2, 7, 7, 5])
    deconv_op = model_util.passthrough(4, 2, True, original_op)
    deconv, = self.eval([deconv_op])

    for row in range(16):
      for column in range(16):
        if row == 0 or row == 15 or column == 0 or column == 15:
          data = deconv[:, row, column, :].flatten()
          # We should have padded with ones.
          self.assertEqual(data.min(), 1)
          self.assertEqual(data.max(), 1)


class ResidualV2ModuleTest(test_util.Base):

  def setUp(self):
    super(ResidualV2ModuleTest, self).setUp()

    original_op = tf.to_float(tf.range(0, 2 * 7 * 7 * 5))
    self.original_op = tf.reshape(original_op, [2, 7, 7, 5])

    self.deconv_op = model_util.passthrough(4, 2, True, self.original_op)
    self.conv_op = model_util.passthrough(4, 2, False, self.deconv_op)

  def test_name(self):
    self.assertRegexpMatches(self.deconv_op.name, 'passthrough')
    self.assertRegexpMatches(self.conv_op.name, 'passthrough')

  def test_size(self):
    self.assertListEqual(self.deconv_op.shape_as_list(), [2, 16, 16, 5])

  def test_bijection(self):
    self.assert_tensors_equal(self.original_op, self.conv_op)

  def test_padding(self):
    original_op = tf.ones([2, 7, 7, 5])
    deconv_op = model_util.passthrough(4, 2, True, original_op)
    deconv, = self.eval([deconv_op])

    for row in range(16):
      for column in range(16):
        if row == 0 or row == 15 or column == 0 or column == 15:
          data = deconv[:, row, column, :].flatten()
          # We should have padded with ones.
          self.assertEqual(data.min(), 1)
          self.assertEqual(data.max(), 1)


class ModuleTest(test_util.Base):

  def setUp(self):
    super(ModuleTest, self).setUp()

    input_op = tf.zeros((7, 32, 32, 5))
    self.module_op = model_util.module(
        3,
        1,
        26,
        8,
        is_deconv=False,
        add_bias=False,
        min_depth_from_residual=False,
        is_train=True,
        input_op=input_op)
    self.residual_v2_module_op = model_util.module(
        3,
        1,
        26,
        8,
        is_deconv=False,
        add_bias=False,
        min_depth_from_residual=False,
        is_train=True,
        input_op=input_op)

  def test_name(self):
    self.assertRegexpMatches(self.module_op.name, 'module')
    self.assertRegexpMatches(self.residual_v2_module_op.name, 'module')

  def test_size(self):
    self.assertListEqual([7, 30, 30, 8], self.module_op.shape_as_list())
    self.assertListEqual([7, 30, 30, 8],
                         self.residual_v2_module_op.shape_as_list())

  def test_forward_pass(self):
    self.eval([self.module_op, self.residual_v2_module_op])


class ModelTest(test_util.Base):

  def setUp(self):
    super(ModelTest, self).setUp()

    core_model = functools.partial(fovea.core, 50)
    add_head = model_util.add_head
    add_head = functools.partial(model_util.add_head, is_residual_conv=True)

    pp = model_util.PredictionParameters(
        [(self.minception_input_lt.axes['z'].labels,
          self.minception_input_lt.axes['channel'].labels),
         (self.minception_target_lt.axes['z'].labels,
          self.minception_target_lt.axes['channel'].labels)], 16)

    [self.diagnostic_prediction_lt, self.prediction_lt] = controller.model(
        core_model, add_head, pp, True, self.minception_input_lt)

  def test_name(self):
    self.assertRegexpMatches(self.diagnostic_prediction_lt.name, 'model.*/0')
    self.assertRegexpMatches(self.prediction_lt.name, 'model.*/1')

  def _test_diagnostic_axes(self):
    golden_axes = lt.Axes([
        self.minception_input_lt.axes['batch'], ('row', 8), ('column', 8),
        self.minception_input_lt.axes['z'],
        self.minception_input_lt.axes['channel'], ('class', 16)
    ])
    self.assertEqual(self.diagnostic_prediction_lt.axes, golden_axes)

  def _test_target_axes(self):
    golden_axes = lt.Axes([
        self.minception_target_lt.axes['batch'], ('row', 8), ('column', 8),
        self.minception_target_lt.axes['z'],
        self.minception_target_lt.axes['channel'], ('class', 16)
    ])
    self.assertEqual(self.prediction_lt.axes, golden_axes)

  def _test_forward_pass(self):
    self.eval([self.diagnostic_prediction_lt, self.prediction_lt])


if __name__ == '__main__':
  test.main()
