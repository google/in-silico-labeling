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

import tensorflow as tf

# pylint: disable=g-bad-import-order
from isl import test_util
from isl import util

flags = tf.flags
test = tf.test
lt = tf.contrib.labeled_tensor

FLAGS = flags.FLAGS


class OnehotTest(test_util.Base):

  def setUp(self):
    super(OnehotTest, self).setUp()

    original_op = tf.to_float(tf.constant(range(4))) / 3.0
    original_lt = lt.LabeledTensor(original_op, ['batch'])
    self.onehot_lt = util.onehot(4, original_lt)

  def test_name(self):
    self.assertIn('onehot', self.onehot_lt.name)

  def test(self):
    golden_lt = lt.LabeledTensor(
        tf.constant([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
        [('batch', 4), ('class', 4)])
    self.assert_labeled_tensors_equal(self.onehot_lt, golden_lt)


class CropCenterTest(test_util.Base):

  def setUp(self):
    super(CropCenterTest, self).setUp()

    original_op = tf.constant([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11],
                               [12, 13, 14, 15]])
    original_op = tf.reshape(original_op, [1, 4, 4, 1])
    original_lt = lt.LabeledTensor(original_op,
                                   ['batch', 'row', 'column', 'channel'])

    self.crop_lt = util.crop_center(2, original_lt)

  def test_name(self):
    self.assertIn('crop_center', self.crop_lt.name)

  def test(self):
    golden_lt = lt.LabeledTensor(
        tf.reshape(tf.constant([[5, 6], [9, 10]]), [1, 2, 2, 1]),
        [('batch', 1), ('row', 2), ('column', 2), ('channel', 1)])
    self.assert_labeled_tensors_equal(self.crop_lt, golden_lt)


class PadConstantTest(test_util.Base):

  def setUp(self):
    super(PadConstantTest, self).setUp()

    self.small_lt = lt.LabeledTensor(tf.constant([1]), axes=['x'])

    self.pad_lt = util.pad_constant(self.small_lt, {'x': (1, 1)}, 42)

  def test_name(self):
    self.assertIn('pad_constant', self.pad_lt.name)

  def test(self):
    golden_lt = lt.LabeledTensor(tf.constant([42, 1, 42]), ['x'])
    self.assert_labeled_tensors_equal(self.pad_lt, golden_lt)


if __name__ == '__main__':
  test.main()
