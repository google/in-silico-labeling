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
"""Convenience wrappers for ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from typing import Optional, Tuple

# pylint: disable=g-bad-import-order
from isl import tensorcheck

logging = tf.logging
lt = tf.contrib.labeled_tensor


def _num_extracted_rows_and_columns(
    image_size: int,
    patch_size: int,
    stride: int,
    num_scales: int,
    scale_factor: int,
) -> int:
  """The number of rows or columns in a patch extraction grid."""
  largest_patch_size = int(patch_size * (scale_factor**(num_scales - 1)))
  residual = image_size - largest_patch_size
  return (residual // stride) + 1


@tensorcheck.well_defined()
def extract_patches_single_scale(
    patch_size: int,
    stride: int,
    image_lt: lt.LabeledTensor,
    name: str = None,
) -> Tuple[np.ndarray, lt.LabeledTensor]:
  """Extract single scale patches from the given image.

  This op is implemented purely in Tensorflow and so may be more memory
  efficient when only single-scale patches are required.

  Args:
    patch_size: The size (width and height) of the extracted patches, assuming
      the patches are square.
    stride: The distance between the centers of neighboring patches, in
      both the x and y directions.
      Set it to 1 to get all patches from the image.
    image_lt: A LabeledTensor with the axes [batch, row, column, channel].
    name: The op name.

  Returns:
    A numpy float array of shape [num_patches, 2] giving the row and
    column centers.

    A LabeledTensor with axes [batch, patch_row, patch_column, row,
      column, channel].

  Raises:
    ValueError: If num_rows - patch_size + 1 < 1.
  """
  with tf.name_scope(name, "extract_patches_single_scale", [image_lt]) as scope:
    image_lt = lt.transpose(image_lt, ["batch", "row", "column", "channel"])
    image_lt = tensorcheck.bounds(0.0, 1.0, image_lt)

    logging.info("extract_patches_single_scale: Input axes: %s", image_lt.axes)

    batch_size = len(image_lt.axes["batch"])
    num_rows = len(image_lt.axes["row"])
    num_columns = len(image_lt.axes["column"])

    row_offsets = range(0, num_rows - patch_size + 1, stride)
    if not row_offsets:
      raise ValueError("num_rows - patch_size + 1 must be >= 1")
    expected_num_rows = _num_extracted_rows_and_columns(num_rows, patch_size,
                                                        stride, 1, 2)
    assert len(row_offsets) == expected_num_rows, (len(row_offsets),
                                                   expected_num_rows,
                                                   (num_rows, patch_size,
                                                    stride))

    column_offsets = range(0, num_columns - patch_size + 1, stride)
    assert column_offsets
    expected_num_columns = _num_extracted_rows_and_columns(
        num_columns, patch_size, stride, 1, 2)
    assert len(column_offsets) == expected_num_columns, (len(column_offsets),
                                                         expected_num_columns,
                                                         (num_rows, patch_size,
                                                          stride))

    offsets = [(r, c) for r in row_offsets for c in column_offsets]

    patch_lts = []
    for b in range(batch_size):
      for (row, column) in offsets:
        patch_lt = lt.slice(
            image_lt, {
                "batch": slice(b, b + 1),
                "row": slice(row, row + patch_size),
                "column": slice(column, column + patch_size)
            })
        patch_lts.append(patch_lt)

    pack_lt = lt.concat(patch_lts, "batch")
    reshape_lt = lt.reshape(pack_lt, ["batch"], [
        image_lt.axes["batch"], ("patch_row", len(row_offsets)),
        ("patch_column", len(column_offsets))
    ])

    reshape_lt = tensorcheck.shape(reshape_lt)
    reshape_lt = tensorcheck.bounds(0.0, 1.0, reshape_lt, name=scope)

    centers = [
        (r + patch_size / 2.0, c + patch_size / 2.0) for (r, c) in offsets
    ]

    logging.info("extract_patches_single_scale: Output axes: %s",
                 reshape_lt.axes)

    return np.array(centers), reshape_lt


def _distribution_statistics(distribution: tf.Tensor) -> tf.Tensor:
  """Implementation of `distribution_statisticsy`."""
  _, num_classes = distribution.shape.as_list()
  assert num_classes is not None

  # Each batch element is a probability distribution.
  max_discrepancy = tf.reduce_max(
      tf.abs(tf.reduce_sum(distribution, axis=1) - 1.0))
  with tf.control_dependencies([tf.assert_less(max_discrepancy, 0.0001)]):
    values = tf.reshape(tf.linspace(0.0, 1.0, num_classes), [1, num_classes])

    mode = tf.to_float(tf.argmax(distribution,
                                 axis=1)) / tf.constant(num_classes - 1.0)
    median = tf.reduce_sum(
        tf.to_float(tf.cumsum(distribution, axis=1) < 0.5),
        axis=1) / tf.constant(num_classes - 1.0)
    mean = tf.reduce_sum(distribution * values, axis=1)
    standard_deviation = tf.sqrt(
        tf.reduce_sum(
            ((values - tf.reshape(mean, [-1, 1]))**2) * distribution, axis=1))
    probability_nonzero = 1.0 - distribution[:, 0]
    entropy = tf.reduce_sum(
        -(distribution * tf.log(distribution + 0.0000001)), axis=1) / tf.log(
            float(num_classes))

    statistics = tf.stack(
        [mode, median, mean, standard_deviation, probability_nonzero, entropy],
        axis=1)

    return statistics


@tensorcheck.well_defined()
def distribution_statistics(distribution_lt: lt.LabeledTensor,
                            name: str = None) -> lt.LabeledTensor:
  """Compute statistics of the given distributions.

  The statistics are computed assuming the first class has value 0.0 and
  the last class has value 1.0, with a linear progression in between.

  Args:
    distribution_lt: A float32 Tensor with axes [batch, class],
      where each row is a probability distribution over the classes.
    name: The op name.

  Returns:
    A float32 Tensor of shape batch_size x 6, containing the following
    statistics:
    1) mode,
    2) median,
    3) mean,
    4) standard deviation,
    5) probability nonzero (the probability assigned to all classes except
      the first), and
    6) discrete entropy ranging from 0.0 (no entropy) to 1.0 (maximum possible
      entropy).
  """
  # This is the number of statistics computed by distribution_statistics.
  num_statistics = 6
  with tf.name_scope(name, "distribution_statistics",
                     [distribution_lt]) as scope:
    distribution_lt = lt.transpose(distribution_lt, ["batch", "class"])

    statistics_op = _distribution_statistics(distribution_lt.tensor)

    statistics_op = tf.reshape(statistics_op, [-1, num_statistics], name=scope)

    statistic_axis = lt.Axis("statistic", [
        "mode", "median", "mean", "standard_deviation", "probability_nonzero",
        "entropy"
    ])
    return lt.LabeledTensor(statistics_op,
                            [distribution_lt.axes["batch"], statistic_axis])


def _num_assembled_rows_and_columns(num_extracted: int, stride: int,
                                    output_size: int) -> int:
  """Helper for alpha_composite."""
  return (num_extracted - 1) * stride + output_size


def patches_to_image(patch_centers: np.ndarray,
                     patch_lt: lt.LabeledTensor,
                     name: Optional[str] = None) -> lt.LabeledTensor:
  """Composite images into a larger image.

  Args:
    patch_centers: The patch centers as a numpy array of size [num_patches, 2],
      where each row is a patch's (row_center, column_center) in the original
      image.
    patch_lt: A float32 tensor with axes [batch, row, column, channel].
      The batch dimension size must be divisible by num_patches.
    name: Optional op name.

  Returns:
    A composited image as a float32 Tensor with axes [batch, row, column,
      channel].
  """
  with tf.name_scope(name, "patches_to_image", [patch_lt]) as scope:
    patch_lt = lt.transpose(patch_lt, ["batch", "row", "column", "channel"])

    num_extracted_rows = len(set([l[0] for l in patch_centers]))
    num_extracted_columns = len(set([l[1] for l in patch_centers]))
    assert num_extracted_rows * num_extracted_columns == patch_centers.shape[0]

    batch_size = len(patch_lt.axes["batch"])
    assert batch_size % len(patch_centers) == 0
    output_batch_size = batch_size // len(patch_centers)

    # TODO(ericmc): This will fail if the stride is not homogeneous.
    if patch_centers.shape[0] == 1:
      stride = 0
    else:
      [row_0, column_0] = patch_centers[0]
      [row_1, column_1] = patch_centers[1]
      if row_0 == row_1:
        stride = column_1 - column_0
      else:
        stride = row_1 - row_0
      assert stride > 0
      assert abs(round(stride) - stride) < 0.0001
      stride = int(round(stride))

    patch_lt = lt.reshape(patch_lt, ["batch"],
                          [("batch", output_batch_size),
                           ("patch_row", num_extracted_rows),
                           ("patch_column", num_extracted_columns)])
    tf.logging.info("%r", patch_lt.axes)

    row_lts = []
    for r in range(num_extracted_rows):
      this_row = []
      for c in range(num_extracted_columns):
        this_row.append(patch_lt[:, r, c, :, :, :])
      row_lts.append(lt.concat(this_row, "column"))
    stitched = lt.concat(row_lts, "row")

    return lt.identity(stitched, name=scope)
