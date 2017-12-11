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

import os

import numpy as np
import tensorflow as tf

# pylint: disable=g-bad-import-order
from isl import tensorcheck
from isl import test_util

lt = tf.contrib.labeled_tensor

FLAGS = tf.flags.FLAGS


class BoundsTest(test_util.Base):

  def setUp(self):
    super(BoundsTest, self).setUp()

    self.one_lt = lt.LabeledTensor(tf.constant(1.0), [])

    self.okay_lt = tensorcheck.bounds(0.5, 1.5, self.one_lt)
    self.lower_error_lt = tensorcheck.bounds(1.1, 1.2, self.one_lt)
    self.upper_error_lt = tensorcheck.bounds(0.5, 0.9, self.one_lt)

  def test_okay(self):
    self.assert_labeled_tensors_equal(self.okay_lt, self.one_lt)

  def test_error(self):
    with self.assertRaisesRegexp(tf.errors.InvalidArgumentError,
                                 'Condition x >= 0 did not hold element-wise'):
      self.eval([self.lower_error_lt])
    with self.assertRaisesRegexp(tf.errors.InvalidArgumentError,
                                 'Condition x <= 0 did not hold element-wise'):
      self.eval([self.upper_error_lt])


class ShapeTest(test_util.Base):

  def setUp(self):
    super(ShapeTest, self).setUp()

    filename_op = tf.train.string_input_producer([
        os.path.join(os.environ['TEST_SRCDIR'],
                     'isl/testdata/research_logo.jpg')
    ])

    reader = tf.WholeFileReader()
    _, encoded_image_op = reader.read(filename_op)
    image_op = tf.image.decode_jpeg(encoded_image_op, channels=3)

    self.correct_shape_op = tf.identity(image_op)
    self.correct_shape_op.set_shape([250, 250, 3])
    self.correct_lt = lt.LabeledTensor(self.correct_shape_op,
                                       ['x', 'y', 'color'])

    self.incorrect_shape_op = tf.identity(image_op)
    self.incorrect_shape_op.set_shape([50, 50, 3])
    self.incorrect_lt = lt.LabeledTensor(self.incorrect_shape_op,
                                         ['x', 'y', 'color'])

    self.okay_lt = tensorcheck.shape(self.correct_lt)
    self.error_lt = tensorcheck.shape(self.incorrect_lt)

  def test_okay(self):
    self.assert_labeled_tensors_equal(self.okay_lt, self.correct_lt)

  def test_error(self):
    with self.assertRaisesRegexp(
        tf.errors.InvalidArgumentError,
        'Runtime shape does not match shape at graph construction time.'):
      self.eval([self.error_lt])


class WellDefinedTest(test_util.Base):

  @tensorcheck.well_defined()
  def add(self, x, labeled_tensor):
    return labeled_tensor + x

  def setUp(self):
    super(WellDefinedTest, self).setUp()

    self.finite_lt = lt.LabeledTensor(tf.constant(42.0), [])
    self.nan_lt = lt.LabeledTensor(tf.constant(np.nan), [])

    self.checked_finite_lt = self.add(1.0, self.finite_lt)
    self.checked_nan_lt = self.add(1.0, self.nan_lt)

  def test_finite(self):
    self.assert_labeled_tensors_equal(self.finite_lt + 1.0,
                                      self.checked_finite_lt)

  def test_nan(self):
    with self.test_session() as sess:
      with self.assertRaisesRegexp(tf.errors.InvalidArgumentError,
                                   'Tensor had NaN values'):
        sess.run([self.checked_nan_lt])


if __name__ == '__main__':
  tf.test.main()
