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

import numpy as np
import tensorflow as tf

# pylint: disable=g-bad-import-order
from isl import data_provider
from isl import test_util
from isl import util

flags = tf.flags
lt = tf.contrib.labeled_tensor
test = tf.test

FLAGS = flags.FLAGS

INPUT_Z_VALUES = [round(v, 4) for v in np.linspace(0.0, 1.0, 13)]
INPUT_CHANNEL_VALUES = ['BRIGHTFIELD', 'PHASE_CONTRAST', 'DIC']
TARGET_Z_VALUES = ['MAXPROJECT']
TARGET_CHANNEL_VALUES = [
    'DAPI_CONFOCAL',
    'DAPI_WIDEFIELD',
    'CELLMASK_CONFOCAL',
    'TUJ1_WIDEFIELD',
    'NFH_CONFOCAL',
    'MAP2_CONFOCAL',
    'ISLET_WIDEFIELD',
    'DEAD_CONFOCAL',
]


class ReadSerializedExampleTest(test_util.Base):

  def setUp(self):
    super(ReadSerializedExampleTest, self).setUp()

    self.example_lt = data_provider.read_serialized_example(
        [self.recordio_path()], True,
        util.BatchParameters(4, num_threads=1, capacity=2), True)

  def test_name(self):
    self.assertIn('read_serialized_example', self.example_lt.name)

  def test_deterministic(self):
    example_0, = self.eval([self.example_lt])
    example_1, = self.eval([self.example_lt])

    self.assertTrue((example_0 == example_1).all())


class ReadPNGsTest(test_util.Base):

  def test_load_image_set(self):
    images = data_provider.load_image_set(
        self.data_path('condition_blue_sample'))
    self.assertEqual(len(images), 16)

  def test_image_set_to_tensor(self):
    images = data_provider.load_image_set(
        self.data_path('condition_blue_sample'))
    tensor = data_provider.image_set_to_tensor(images, INPUT_Z_VALUES,
                                               INPUT_CHANNEL_VALUES, 0, 0, 1024)
    self.assertListEqual(tensor.tensor.shape.as_list(),
                         [1, 1024, 1024, 13, 3, 2])


class CroppedInputAndTargetTest(test_util.Base):

  def setUp(self):
    super(CroppedInputAndTargetTest, self).setUp()

    batch_size = 2
    rtp = data_provider.ReadTableParameters(
        shard_paths=[self.recordio_path()],
        is_recordio=True,
        bp=util.BatchParameters(batch_size, num_threads=1, capacity=2),
        is_deterministic=True,
        pad_size=0,
        crop_size=512)

    dp = data_provider.DataParameters(
        rtp,
        input_z_values=self.input_z_values,
        input_channel_values=self.input_channel_values,
        target_z_values=self.target_z_values,
        target_channel_values=self.target_channel_values)

    self.input_lt, self.target_lt = data_provider.cropped_input_and_target(dp)

  def test_name(self):
    self.assertRegexpMatches(self.input_lt.name,
                             'cropped_input_and_target.*/input')
    self.assertRegexpMatches(self.target_lt.name,
                             'cropped_input_and_target.*/target')

  def test_deterministic(self):
    [input_0, target_0] = self.eval([self.input_lt, self.target_lt])
    [input_1, target_1] = self.eval([self.input_lt, self.target_lt])

    self.assertTrue((input_0 == input_1).all())
    self.assertTrue((target_0 == target_1).all())

  def test_batch_contains_different_examples(self):
    input_0_lt = lt.slice(self.input_lt, {'batch': 0})
    input_1_lt = lt.slice(self.input_lt, {'batch': 1})
    target_0_lt = lt.slice(self.target_lt, {'batch': 0})
    target_1_lt = lt.slice(self.target_lt, {'batch': 1})
    [input_0, input_1, target_0,
     target_1] = self.eval([input_0_lt, input_1_lt, target_0_lt, target_1_lt])

    self.assertFalse((input_0 == input_1).all())
    self.assertFalse((target_0 == target_1).all())

  def test(self):
    self.assertListEqual(self.input_lt.axes.keys(), util.CANONICAL_AXIS_ORDER)
    self.assertListEqual(self.target_lt.axes.keys(), util.CANONICAL_AXIS_ORDER)

    self.save_images('cropped_input_and_target', [
        self.get_images('', self.input_lt),
        self.get_images('', self.target_lt)
    ])
    self.assert_images_near('cropped_input_and_target', only_check_size=True)


if __name__ == '__main__':
  test.main()
