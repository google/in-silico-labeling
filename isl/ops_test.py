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
import ops
import test_util

flags = tf.flags
test = tf.test
lt = tf.contrib.labeled_tensor
gfile = tf.gfile

FLAGS = flags.FLAGS


class ExtractPatchesSingleScaleTest(test_util.Base):

  def setUp(self):
    super(ExtractPatchesSingleScaleTest, self).setUp()

    concat_lt = lt.concat([self.load_map2(), self.load_nfh()], "channel")

    def slice_concat(offset):
      return lt.slice(concat_lt, {
          "row": slice(offset, offset + 400),
          "column": slice(offset, offset + 400)
      })

    slice_0_lt = slice_concat(0)
    slice_1_lt = slice_concat(400)
    batch_lt = lt.concat([slice_0_lt, slice_1_lt], "batch")

    _, self.patch_lt = ops.extract_patches_single_scale(256, 64, batch_lt)

  def test_name(self):
    self.assertIn("extract_patches_single_scale", self.patch_lt.name)

  def test_deterministic(self):
    [patch_0] = self.eval([self.patch_lt])
    [patch_1] = self.eval([self.patch_lt])
    self.assertTrue((patch_0 == patch_1).all())

  def test(self):
    self.assertEqual(self.patch_lt.axes,
                     lt.Axes([("batch", 2), ("patch_row", 3),
                              ("patch_column", 3), ("row", 256), ("column",
                                                                  256),
                              ("channel", ["MAP2_CONFOCAL", "NFH_CONFOCAL"])]))

    self.save_images("extract_patches_single_scale",
                     [self.get_images("", self.patch_lt)])
    self.assert_images_near("extract_patches_single_scale")


class DistributionStatisticsTest(test_util.Base):

  def setUp(self):
    super(DistributionStatisticsTest, self).setUp()

    probability_tensor = tf.constant([
        [0, 1, 0, 0],
        [0.25, 0.25, 0.25, 0.25],
        [0, 0.5, 0, 0.5],
        [0.1, 0.2, 0.2, 0.5],
    ])
    probability_lt = lt.LabeledTensor(probability_tensor, [("batch", 4),
                                                           ("class", 4)])
    self.statistic_lt = ops.distribution_statistics(probability_lt)

  def test_name(self):
    self.assertIn("distribution_statistics", self.statistic_lt.name)

  def test(self):
    golden_axes = lt.Axes([("batch", 4), ("statistic", [
        "mode", "median", "mean", "standard_deviation", "probability_nonzero",
        "entropy"
    ])])
    self.assertEqual(self.statistic_lt.axes, golden_axes)

    [distribution_statistics] = self.eval([self.statistic_lt.tensor])
    # Compare values.
    golden = np.array([
        [0.33333334, 0.33333334, 0.33333334, 0., 1., 0.0],
        [0.0, 0.33333334, 0.5, 0.37267799624996495, 0.75, 1.0],
        [0.33333334, 0.33333334, 0.66666669, 0.3333333, 1., 0.5],
        [1., 0.66666669, 0.7, 0.34801021, 0.89999998, 0.8804820237218407],
    ])
    np.testing.assert_allclose(
        golden, distribution_statistics, atol=1e-5, rtol=1e-5)


class PatchesToImageTest(test_util.Base):

  def setUp(self):
    super(PatchesToImageTest, self).setUp()

    concat_lt = lt.concat([self.load_map2(), self.load_nfh()], "channel")

    def slice_concat(offset):
      return lt.slice(concat_lt, {
          "row": slice(offset, offset + 400),
          "column": slice(offset, offset + 400)
      })

    slice_0_lt = slice_concat(0)
    slice_1_lt = slice_concat(400)
    batch_lt = lt.concat([slice_0_lt, slice_1_lt], "batch")

    centers, patch_lt = ops.extract_patches_single_scale(64, 64, batch_lt)

    self.composite_lt = ops.patches_to_image(
        centers,
        lt.reshape(patch_lt, ["batch", "patch_row", "patch_column"], ["batch"]))

  def test_name(self):
    self.assertIn("patches_to_image", self.composite_lt.name)

  def test(self):
    self.assertEqual(self.composite_lt.axes,
                     lt.Axes([("batch", 2), ("row", 384), ("column", 384),
                              ("channel", ["MAP2_CONFOCAL", "NFH_CONFOCAL"])]))

    self.save_images("patches_to_image",
                     [self.get_images("", self.composite_lt)])
    self.assert_images_near("patches_to_image")


class TrainTupleToImagesTest(test_util.Base):

  def setUp(self):
    super(TrainTupleToImagesTest, self).setUp()

    input_z_values = [round(v, 4) for v in np.linspace(0.0, 1.0, 140)]
    input_channel_values = [
        "BRIGHTFIELD", "PHASE_CONTRAST", "DIC", "HPAM", "PHENOSCAPE_INPUT"
    ]
    target_z_values = ["MAXPROJECT"]
    target_channel_values = [
        "DAPI_CONFOCAL",
        "DAPI_WIDEFIELD",
        "CELLMASK_CONFOCAL",
        "TUJ1_WIDEFIELD",
        "NFH_CONFOCAL",
        "MAP2_CONFOCAL",
        "ISLET_WIDEFIELD",
        "DEAD_CONFOCAL",
        "CENTER",
        "RED",
        "GREEN",
        "BLUE",
    ]

    tensorflow_example_path = self.data_path(
        "train_tuple_to_images/train_tuple_tensorflow_example.pb")
    encoded_example = gfile.GFile(tensorflow_example_path).read()

    encoded_example_op = tf.constant(encoded_example)
    features = {"TRAIN_TUPLE": tf.FixedLenFeature([], dtype=tf.string)}
    encoded_train_tuple_op = tf.reshape(
        tf.parse_example(tf.reshape(encoded_example_op, [1]),
                         features)["TRAIN_TUPLE"], [])

    self.encoded_image_op = ops.train_tuple_to_images(
        [str(z) for z in input_z_values] + target_z_values,
        input_channel_values + target_channel_values, [False, True],
        encoded_train_tuple_op)

    self.expected_shape = (140 + 1, 5 + 12, 2)

  def test_name(self):
    self.assertIn("train_tuple_to_images", self.encoded_image_op.name)

  def test(self):
    encoded_images, = self.eval([self.encoded_image_op])
    self.assertEqual(encoded_images.shape, self.expected_shape)


if __name__ == "__main__":
  test.main()
