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
from isl import augment
from isl import controller
from isl import data_provider
from isl import test_util
from isl import util
from isl.models import fovea
from isl.models import model_util

flags = tf.flags
test = tf.test
lt = tf.contrib.labeled_tensor

FLAGS = flags.FLAGS


class Base(test_util.Base):

  def setUp(self):
    super(Base, self).setUp()

    rtp = data_provider.ReadTableParameters(
        shard_paths=[self.recordio_path()],
        is_recordio=True,
        bp=util.BatchParameters(2, num_threads=1, capacity=2),
        is_deterministic=True,
        pad_size=51,
        crop_size=110 + 24)
    self.dp = data_provider.DataParameters(
        rtp,
        input_z_values=self.input_z_values,
        input_channel_values=self.input_channel_values,
        target_z_values=self.target_z_values,
        target_channel_values=self.target_channel_values)

    self.ap = augment.AugmentParameters(
        offset_standard_deviation=0.1,
        multiplier_standard_deviation=0.05,
        noise_standard_deviation=0.1)

    self.extract_patch_size = 80
    self.stride = 8
    self.stitch_patch_size = 8

    self.bp = util.BatchParameters(4, 1, 4)

    self.core_model = functools.partial(fovea.core, 50)
    self.add_head = functools.partial(
        model_util.add_head, is_residual_conv=True)

    self.num_classes = 16
    self.shuffle = False


class ProvidePreprocessedDataTest(Base):

  def setUp(self):
    super(ProvidePreprocessedDataTest, self).setUp()

    (self.patch_centers, self.input_patch_lt,
     self.target_patch_lt) = controller.provide_preprocessed_data(
         self.dp, self.ap, self.extract_patch_size, self.stride)

  def test_name(self):
    self.assertRegexpMatches(self.input_patch_lt.name,
                             'provide_preprocessed_data.*/input')
    self.assertRegexpMatches(self.target_patch_lt.name,
                             'provide_preprocessed_data.*/target')

  def test_input_axes(self):
    golden_axes = lt.Axes(
        [('batch', 98), ('row', 80), ('column', 80), self.input_lt.axes['z'],
         self.input_lt.axes['channel'], ('mask', [False, True])])
    self.assertEqual(self.input_patch_lt.axes, golden_axes)

  def test_target_axes(self):
    golden_axes = lt.Axes(
        [('batch', 98), ('row', 80), ('column', 80), self.target_lt.axes['z'],
         self.target_lt.axes['channel'], ('mask', [False, True])])
    self.assertEqual(self.target_patch_lt.axes, golden_axes)

  def test(self):
    self.save_images('provide_preprocessed_data', [
        self.get_images('input', self.input_patch_lt),
        self.get_images('target', self.target_patch_lt)
    ])
    self.assert_images_near('provide_preprocessed_data', True)


class GetInputTargetAndPredictedTest(Base):

  def setUp(self):
    super(GetInputTargetAndPredictedTest, self).setUp()

    is_train = True

    gitapp = controller.GetInputTargetAndPredictedParameters(
        self.dp, self.ap, 110, self.stride, self.stitch_patch_size, self.bp,
        self.core_model, self.add_head, self.shuffle, self.num_classes,
        util.softmax_cross_entropy, is_train)

    (self.patch_centers, self.input_patch_lt, self.target_patch_lt,
     self.predicted_input_lt, self.predicted_target_lt
    ) = controller.get_input_target_and_predicted(gitapp)

  def test_name(self):
    self.assertRegexpMatches(self.input_patch_lt.name,
                             'get_input_target_and_predicted.*/input')
    self.assertRegexpMatches(self.target_patch_lt.name,
                             'get_input_target_and_predicted.*/target')
    self.assertRegexpMatches(self.predicted_input_lt.name,
                             'get_input_target_and_predicted.*/predict_input')
    self.assertRegexpMatches(self.predicted_target_lt.name,
                             'get_input_target_and_predicted.*/predict_target')

  def test_predict_input_axes(self):
    golden_axes = lt.Axes(
        [('batch', 4), ('row', 8), ('column', 8), self.input_patch_lt.axes['z'],
         self.input_patch_lt.axes['channel'], ('class', self.num_classes)])
    self.assertEqual(self.predicted_input_lt.axes, golden_axes)

  def test_predict_target_axes(self):
    golden_axes = lt.Axes(
        [('batch', 4), ('row', 8), ('column',
                                    8), self.target_patch_lt.axes['z'],
         self.target_patch_lt.axes['channel'], ('class', self.num_classes)])
    self.assertEqual(self.predicted_target_lt.axes, golden_axes)


class SetupLossesTest(Base):

  def setUp(self):
    super(SetupLossesTest, self).setUp()

    is_train = True

    gitapp = controller.GetInputTargetAndPredictedParameters(
        self.dp, self.ap, 110, self.stride, self.stitch_patch_size, self.bp,
        self.core_model, self.add_head, self.shuffle, self.num_classes,
        util.softmax_cross_entropy, is_train)

    (self.input_loss_lts,
     self.target_loss_lts) = controller.setup_losses(gitapp)

  def test_input_keys(self):
    keys = list(sorted(set(self.input_loss_lts.keys())))
    golden_keys = sorted(
        ['%r/TRANSMISSION' % round(z, 4) for z in self.input_z_values])

    self.assertListEqual(keys, golden_keys)

  def test_target_keys(self):
    keys = list(sorted(set(self.target_loss_lts.keys())))
    golden_keys = sorted([
        'MAXPROJECT/%s' % channel
        for channel in self.target_channel_values + ['NEURITE_CONFOCAL']
    ])

    self.assertListEqual(keys, golden_keys)

  def test(self):

    def mean(lts):
      sum_op = tf.add_n([t.tensor for t in lts.values()])
      return sum_op / float(len(lts))

    loss_op = (mean(self.input_loss_lts) + mean(self.target_loss_lts)) / 2.0

    train_op = tf.train.GradientDescentOptimizer(0.00001).minimize(loss_op)
    self.eval([train_op])


class SetupStitchTest(Base):

  def setUp(self):
    super(SetupStitchTest, self).setUp()

    dp = self.dp._replace(
        io_parameters=self.dp.io_parameters._replace(crop_size=110 + 128))
    # self.dp.io_parameters.crop_size = 110 + 128
    is_train = True

    gitapp = controller.GetInputTargetAndPredictedParameters(
        dp, self.ap, 110, self.stride, self.stitch_patch_size, self.bp,
        self.core_model, self.add_head, self.shuffle, self.num_classes,
        util.softmax_cross_entropy, is_train)

    self.image_lt_dict = controller.setup_stitch(gitapp)

  def test_name(self):
    self.assertRegexpMatches(self.image_lt_dict['input_error_panel'].name,
                             'setup_stitch.*/input_error_panel')
    self.assertRegexpMatches(self.image_lt_dict['target_error_panel'].name,
                             'setup_stitch.*/target_error_panel')

  def test(self):
    filenames_and_images = []
    for name, image_lt in self.image_lt_dict.items():
      if name == 'input_error_panel' or name == 'target_error_panel':
        filenames_and_images.append(self.get_images(name + '_', image_lt))

    self.save_images('setup_stitch', filenames_and_images)
    self.assert_images_near('setup_stitch', True)


if __name__ == '__main__':
  test.main()
