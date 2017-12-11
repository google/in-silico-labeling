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
from isl import augment
from isl import test_util
from isl import util

flags = tf.flags
test = tf.test
lt = tf.contrib.labeled_tensor

FLAGS = flags.FLAGS


class CorruptTest(test_util.Base):

  def setUp(self):
    super(CorruptTest, self).setUp()

    self.signal_lt = lt.select(self.input_lt, {'mask': util.slice_1(False)})
    rc = lt.ReshapeCoder(['z', 'channel', 'mask'], ['channel'])
    self.corrupt_coded_lt = augment.corrupt(0.1, 0.05, 0.1,
                                            rc.encode(self.signal_lt))
    self.corrupt_lt = rc.decode(self.corrupt_coded_lt)

  def test_name(self):
    self.assertIn('corrupt', self.corrupt_coded_lt.name)

  def test(self):
    self.assertEqual(self.corrupt_lt.axes, self.signal_lt.axes)

    self.save_images('corrupt', [self.get_images('', self.corrupt_lt)])
    self.assert_images_near('corrupt', True)


class AugmentTest(test_util.Base):

  def setUp(self):
    super(AugmentTest, self).setUp()

    ap = augment.AugmentParameters(0.1, 0.05, 0.1)

    self.input_augment_lt, self.target_augment_lt = augment.augment(
        ap, self.input_lt, self.target_lt)

  def test_name(self):
    self.assertIn('augment/input', self.input_augment_lt.name)
    self.assertIn('augment/target', self.target_augment_lt.name)

  def test(self):
    self.assertEqual(self.input_augment_lt.axes, self.input_lt.axes)
    self.assertEqual(self.target_augment_lt.axes, self.target_lt.axes)

    self.save_images('augment', [
        self.get_images('input_', self.input_augment_lt),
        self.get_images('target_', self.target_augment_lt)
    ])
    self.assert_images_near('augment', True)


if __name__ == '__main__':
  test.main()
