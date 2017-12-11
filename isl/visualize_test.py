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
from isl import data_provider
from isl import test_util
from isl import util
from isl import visualize

flags = tf.flags
test = tf.test
lt = tf.contrib.labeled_tensor

FLAGS = flags.FLAGS


class ErrorBase(test_util.Base):

  def setUp(self):
    super(ErrorBase, self).setUp()

    rtp = data_provider.ReadTableParameters([self.recordio_path()], True,
                                            util.BatchParameters(2, 1, 2), True,
                                            0, 768)
    dp = data_provider.DataParameters(
        rtp, self.input_z_values, self.input_channel_values,
        self.target_z_values, self.target_channel_values)
    _, self.target_lt = data_provider.cropped_input_and_target(dp)

    self.target_lt_0 = lt.select(self.target_lt, {
        'channel': 'DAPI_CONFOCAL',
        'mask': False
    })
    self.target_lt_1 = lt.select(self.target_lt, {
        'channel': 'NEURITE_CONFOCAL',
        'mask': False
    })

    self.target_lt_0 = lt.reshape(self.target_lt_0,
                                  self.target_lt_0.axes.keys()[3:], ['channel'])
    self.target_lt_1 = lt.reshape(self.target_lt_1,
                                  self.target_lt_1.axes.keys()[3:], ['channel'])


class AdditiveErrorTest(ErrorBase):

  def setUp(self):
    super(AdditiveErrorTest, self).setUp()

    self.additive_lt = visualize.additive_error(self.target_lt_0,
                                                self.target_lt_1)

  def test_name(self):
    self.assertIn('additive_error', self.additive_lt.name)

  def test(self):
    self.save_images('additive_error', [
        self.get_images('error_', self.additive_lt),
        self.get_images('target_1_', self.target_lt_1),
        self.get_images('target_0_', self.target_lt_0)
    ])
    self.assert_images_near('additive_error', True)


class SubtractiveErrorTest(ErrorBase):

  def setUp(self):
    super(SubtractiveErrorTest, self).setUp()

    self.subtractive_lt = visualize.subtractive_error(self.target_lt_0,
                                                      self.target_lt_1)

  def test_name(self):
    self.assertIn('subtractive_error', self.subtractive_lt.name)

  def test(self):
    self.save_images('subtractive_error', [
        self.get_images('error_', self.subtractive_lt),
        self.get_images('target_1_', self.target_lt_1),
        self.get_images('target_0_', self.target_lt_0)
    ])
    self.assert_images_near('subtractive_error', True)


class CrossEntropyErrorTest(ErrorBase):

  def setUp(self):
    super(CrossEntropyErrorTest, self).setUp()

    target_lt_0 = util.onehot(16, self.target_lt_0)
    target_lt_1 = util.onehot(16, self.target_lt_1)
    self.ce_lt = visualize.cross_entropy_error(target_lt_0, target_lt_1)

  def test_name(self):
    self.assertIn('cross_entropy_error', self.ce_lt.name)

  def test(self):
    self.save_images('cross_entropy_error', [
        self.get_images('error_', self.ce_lt),
        self.get_images('target_1_', self.target_lt_1),
        self.get_images('target_0_', self.target_lt_0)
    ])
    self.assert_images_near('cross_entropy_error', True)


class ErrorPanelTest(test_util.Base):

  def setUp(self):
    super(ErrorPanelTest, self).setUp()

    rtp = data_provider.ReadTableParameters([self.recordio_path()], True,
                                            util.BatchParameters(2, 1, 2), True,
                                            0, 768)
    dp = data_provider.DataParameters(
        rtp, self.input_z_values, self.input_channel_values,
        self.target_z_values, self.target_channel_values)
    _, batch_target_lt = data_provider.cropped_input_and_target(dp)

    self.prediction_lt = lt.slice(
        lt.select(batch_target_lt, {
            'mask': False
        }), {
            'batch': slice(0, 1)
        })
    self.prediction_lt = util.onehot(16, self.prediction_lt)

    self.target_lt = lt.slice(batch_target_lt, {'batch': slice(1, 2)})

    self.error_panel_lt = visualize.error_panel(self.target_lt,
                                                self.prediction_lt)

  def test_name(self):
    self.assertIn('error_panel', self.error_panel_lt.name)

  def test(self):
    self.save_images('error_panel', [
        self.get_images('error_', self.error_panel_lt),
        self.get_images('target_', self.target_lt),
        self.get_images('prediction_', self.prediction_lt)
    ])
    self.assert_images_near('error_panel', True)


if __name__ == '__main__':
  test.main()
