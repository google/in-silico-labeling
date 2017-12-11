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
"""Library for running inference on a single image."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import tensorflow as tf
from typing import List, Optional

# pylint: disable=g-bad-import-order
from isl import controller
from isl import data_provider
from isl import ops
from isl import util
from isl import visualize

gfile = tf.gfile
logging = tf.logging
lt = tf.contrib.labeled_tensor


def infer(
    gitapp: controller.GetInputTargetAndPredictedParameters,
    restore_directory: str,
    output_directory: str,
    extract_patch_size: int,
    stitch_stride: int,
    infer_size: int,
    channel_whitelist: Optional[List[str]],
    simplify_error_panels: bool,
):
  """Runs inference on an image.

  Args:
    gitapp: GetInputTargetAndPredictedParameters.
    restore_directory: Where to restore the model from.
    output_directory: Where to write the generated images.
    extract_patch_size: The size of input to the model.
    stitch_stride: The stride size when running model inference.
      Equivalently, the output size of the model.
    infer_size: The number of simultaneous inferences to perform in the
      row and column dimensions.
      For example, if this is 8, inference will be performed in 8 x 8 blocks
      for a batch size of 64.
    channel_whitelist: If provided, only images for the given channels will
      be produced.
      This can be used to create simpler error panels.
    simplify_error_panels: Whether to create simplified error panels.

  Raises:
    ValueError: If
      1) The DataParameters don't contain a ReadPNGsParameters.
      2) The images must be larger than the input to the network.
      3) The graph must not contain queues.
  """
  rpp = gitapp.dp.io_parameters
  if not isinstance(rpp, data_provider.ReadPNGsParameters):
    raise ValueError(
        'Data provider must contain a ReadPNGsParameter, but was: %r',
        gitapp.dp)

  original_crop_size = rpp.crop_size
  image_num_rows, image_num_columns = util.image_size(rpp.directory)
  logging.info('Uncropped image size is %d x %d', image_num_rows,
               image_num_columns)
  image_num_rows = min(image_num_rows, original_crop_size)
  if image_num_rows < extract_patch_size:
    raise ValueError(
        'Image is too small for inference to be performed: %d vs %d',
        image_num_rows, extract_patch_size)
  image_num_columns = min(image_num_columns, original_crop_size)
  if image_num_columns < extract_patch_size:
    raise ValueError(
        'Image is too small for inference to be performed: %d vs %d',
        image_num_columns, extract_patch_size)
  logging.info('After cropping, input image size is (%d, %d)', image_num_rows,
               image_num_columns)

  num_row_inferences = (image_num_rows - extract_patch_size) // (
      stitch_stride * infer_size)
  num_column_inferences = (image_num_columns - extract_patch_size) // (
      stitch_stride * infer_size)
  logging.info('Running %d x %d inferences', num_row_inferences,
               num_column_inferences)
  num_output_rows = (num_row_inferences * infer_size * stitch_stride)
  num_output_columns = (num_column_inferences * infer_size * stitch_stride)
  logging.info('Output image size is (%d, %d)', num_output_rows,
               num_output_columns)

  g = tf.Graph()
  with g.as_default():
    row_start = tf.placeholder(dtype=np.int32, shape=[])
    column_start = tf.placeholder(dtype=np.int32, shape=[])
    # Replace the parameters with a new set, which will cause the network to
    # run inference in just a local region.
    gitapp = gitapp._replace(
        dp=gitapp.dp._replace(
            io_parameters=rpp._replace(
                row_start=row_start,
                column_start=column_start,
                crop_size=(infer_size - 1) * stitch_stride + extract_patch_size,
            )))

    visualization_lts = controller.setup_stitch(gitapp)

    def get_statistics(tensor):
      rc = lt.ReshapeCoder(list(tensor.axes.keys())[:-1], ['batch'])
      return rc.decode(ops.distribution_statistics(rc.encode(tensor)))

    visualize_input_lt = visualization_lts['input']
    visualize_predict_input_lt = get_statistics(
        visualization_lts['predict_input'])
    visualize_target_lt = visualization_lts['target']
    visualize_predict_target_lt = get_statistics(
        visualization_lts['predict_target'])

    input_lt = lt.LabeledTensor(
        tf.placeholder(
            dtype=np.float32,
            shape=[
                1, num_output_rows, num_output_columns,
                len(gitapp.dp.input_z_values), 1, 2
            ]),
        axes=[
            'batch',
            'row',
            'column',
            ('z', gitapp.dp.input_z_values),
            ('channel', ['TRANSMISSION']),
            ('mask', [False, True]),
        ])
    predict_input_lt = lt.LabeledTensor(
        tf.placeholder(
            dtype=np.float32,
            shape=[
                1,
                num_output_rows,
                num_output_columns,
                len(gitapp.dp.input_z_values),
                1,
                len(visualize_predict_input_lt.axes['statistic']),
            ]),
        axes=[
            'batch',
            'row',
            'column',
            ('z', gitapp.dp.input_z_values),
            ('channel', ['TRANSMISSION']),
            visualize_predict_input_lt.axes['statistic'],
        ])
    input_error_panel_lt = visualize.error_panel_from_statistics(
        input_lt, predict_input_lt, simplify_error_panels)

    target_lt = lt.LabeledTensor(
        tf.placeholder(
            dtype=np.float32,
            shape=[
                1, num_output_rows, num_output_columns,
                len(gitapp.dp.target_z_values),
                len(gitapp.dp.target_channel_values) + 1, 2
            ]),
        axes=[
            'batch',
            'row',
            'column',
            ('z', gitapp.dp.target_z_values),
            ('channel', gitapp.dp.target_channel_values + ['NEURITE_CONFOCAL']),
            ('mask', [False, True]),
        ])
    predict_target_lt = lt.LabeledTensor(
        tf.placeholder(
            dtype=np.float32,
            shape=[
                1,
                num_output_rows,
                num_output_columns,
                len(gitapp.dp.target_z_values),
                len(gitapp.dp.target_channel_values) + 1,
                len(visualize_predict_target_lt.axes['statistic']),
            ]),
        axes=[
            'batch',
            'row',
            'column',
            ('z', gitapp.dp.target_z_values),
            ('channel', gitapp.dp.target_channel_values + ['NEURITE_CONFOCAL']),
            visualize_predict_target_lt.axes['statistic'],
        ])

    logging.info('input_lt: %r', input_lt)
    logging.info('predict_input_lt: %r', predict_input_lt)
    logging.info('target_lt: %r', target_lt)
    logging.info('predict_target_lt: %r', predict_target_lt)

    def select_channels(tensor):
      if channel_whitelist is not None:
        return lt.select(tensor, {'channel': channel_whitelist})
      else:
        return tensor

    target_error_panel_lt = visualize.error_panel_from_statistics(
        select_channels(target_lt), select_channels(predict_target_lt),
        simplify_error_panels)

    # There shouldn't be any queues in this configuration.
    queue_runners = g.get_collection(tf.GraphKeys.QUEUE_RUNNERS)
    if queue_runners:
      raise ValueError('Graph must not have queues, but had: %r', queue_runners)

    logging.info('Attempting to find restore checkpoint in %s',
                 restore_directory)
    init_fn = util.restore_model(
        restore_directory, restore_logits=True, restore_global_step=True)

    with tf.Session() as sess:
      logging.info('Generating images')
      init_fn(sess)

      input_rows = []
      predict_input_rows = []
      target_rows = []
      predict_target_rows = []
      for infer_row in range(num_row_inferences):
        input_row = []
        predict_input_row = []
        target_row = []
        predict_target_row = []
        for infer_column in range(num_column_inferences):
          rs = infer_row * infer_size * stitch_stride
          cs = infer_column * infer_size * stitch_stride
          logging.info('Running inference at offset: (%d, %d)', rs, cs)
          [inpt, predict_input, target, predict_target] = sess.run(
              [
                  visualize_input_lt,
                  visualize_predict_input_lt,
                  visualize_target_lt,
                  visualize_predict_target_lt,
              ],
              feed_dict={
                  row_start: rs,
                  column_start: cs
              })

          input_row.append(inpt)
          predict_input_row.append(predict_input)
          target_row.append(target)
          predict_target_row.append(predict_target)
        input_rows.append(np.concatenate(input_row, axis=2))
        predict_input_rows.append(np.concatenate(predict_input_row, axis=2))
        target_rows.append(np.concatenate(target_row, axis=2))
        predict_target_rows.append(np.concatenate(predict_target_row, axis=2))

      logging.info('Stitching')
      stitched_input = np.concatenate(input_rows, axis=1)
      stitched_predict_input = np.concatenate(predict_input_rows, axis=1)
      stitched_target = np.concatenate(target_rows, axis=1)
      stitched_predict_target = np.concatenate(predict_target_rows, axis=1)

      logging.info('Creating error panels')
      [input_error_panel, target_error_panel, global_step] = sess.run(
          [
              input_error_panel_lt, target_error_panel_lt,
              tf.train.get_global_step()
          ],
          feed_dict={
              input_lt: stitched_input,
              predict_input_lt: stitched_predict_input,
              target_lt: stitched_target,
              predict_target_lt: stitched_predict_target,
          })

      output_directory = os.path.join(output_directory, '%.8d' % global_step)
      if not gfile.Exists(output_directory):
        gfile.MakeDirs(output_directory)

      util.write_image(
          os.path.join(output_directory, 'input_error_panel.png'),
          input_error_panel[0, :, :, :])
      util.write_image(
          os.path.join(output_directory, 'target_error_panel.png'),
          target_error_panel[0, :, :, :])

      logging.info('Done generating images')
