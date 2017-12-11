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
"""Logic for training and evaluating a Seeing More model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import tensorflow as tf
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple

# pylint: disable=g-bad-import-order
from isl import augment
from isl import data_provider
from isl import ops
from isl import tensorcheck
from isl import util
from isl import visualize
from isl.models import model_util

logging = tf.logging
lt = tf.contrib.labeled_tensor


def provide_preprocessed_data(
    dp: data_provider.DataParameters,
    ap: Optional[augment.AugmentParameters],
    extract_patch_size: int,
    stride: int,
    name: str = None,
) -> Tuple[np.ndarray, lt.LabeledTensor, lt.LabeledTensor]:
  """Provide preprocessed input and target patches.

  Args:
    dp: DataParameters.
    ap: Optional AugmentParameters.
    extract_patch_size: Patch size for patch extraction.
    stride: Stride for patch extraction.
    name: Optional op name.

  Returns:
    An array containing patch center locations.
    The patches are extracted with padding, so that the stitched model outputs
    will form an image the same size as the input.

    A tensor with model inputs, possibly jittered and corrupted for data
    enrichment.

    A tensor with model targets.
  """
  with tf.name_scope(name, 'provide_preprocessed_data', []) as scope:
    input_lt, target_lt = data_provider.cropped_input_and_target(dp)
    visualize.summarize_image(
        visualize.canonical_image(input_lt, name=scope + 'input'))
    visualize.summarize_image(
        visualize.canonical_image(target_lt, name=scope + 'target'))

    if ap is not None:
      input_lt, target_lt = augment.augment(ap, input_lt, target_lt)
      visualize.summarize_image(
          visualize.canonical_image(input_lt, name=scope + 'input_jitter'))
      visualize.summarize_image(
          visualize.canonical_image(target_lt, name=scope + 'target_jitter'))

    rc = lt.ReshapeCoder(['z', 'channel', 'mask'], ['channel'])

    patch_centers, input_lt = ops.extract_patches_single_scale(
        extract_patch_size, stride, rc.encode(input_lt))
    input_lt = rc.decode(input_lt)
    input_lt = lt.reshape(
        input_lt, ['batch', 'patch_row', 'patch_column'], ['batch'],
        name=scope + 'input')

    rc = lt.ReshapeCoder(['z', 'channel', 'mask'], ['channel'])
    target_lt = rc.decode(
        ops.extract_patches_single_scale(extract_patch_size, stride,
                                         rc.encode(target_lt))[1])
    target_lt = lt.reshape(
        target_lt, ['batch', 'patch_row', 'patch_column'], ['batch'],
        name=scope + 'target')

    return patch_centers, input_lt, target_lt


# Parameters for configuring a network and getting outputs.
#
# Args:
#   dp: DataParameters.
#   ap: Optional AugmentParameters.
#   extract_patch_size: Patch extraction size.
#   stride: Patch extraction stride.
#   stitch_patch_size: Image stitching patch size.
#   bp: Batch parameters for rebatch immediately before model is called.
#   core_model: Function which takes an input tensor of shape [batch, row,
#     column, depth] and returns an embedding tensor of shape [batch, row',
#     column', depth'].
#   add_head: Function which takes an embedding layer of shape [batch, row,
#     column, depth] and returns an output head of shape [batch, row, column,
#     num_classes].
#   shuffle: Whether to shuffle the batch.
#   num_classes: Number of pixel classes to predict.
#   loss: The loss function to use.
#   is_train: Whether we're training this graph.
# pylint: disable=invalid-name
GetInputTargetAndPredictedParameters = NamedTuple(
    'GetInputTargetAndPredictedParameters', [
        ('dp', data_provider.DataParameters),
        ('ap', Optional[augment.AugmentParameters]),
        ('extract_patch_size', int),
        ('stride', int),
        ('stitch_patch_size', int),
        ('bp', Optional[util.BatchParameters]),
        ('core_model', Callable),
        ('add_head', Callable),
        ('shuffle', bool),
        ('num_classes', int),
        ('loss', Callable),
        ('is_train', bool),
    ])
# pylint: enable=invalid-name


@tensorcheck.well_defined()
def model(core_model: Callable,
          add_head: Callable,
          pp: model_util.PredictionParameters,
          is_train: bool,
          input_lt: lt.LabeledTensor,
          name: str = None) -> List[lt.LabeledTensor]:
  """Predict targets from canonical input.

  Args:
    core_model: Function which takes an input tensor of shape [batch, row,
      column, depth] and returns an embedding tensor of shape [batch, row',
      column', depth'].
    add_head: Function which takes an embedding layer of shape [batch, row,
      column, depth] and returns an output head of shape [batch, row, column,
      num_classes].
    pp: PredictionParameters.
    is_train: Whether we're training this graph.
    input_lt: The input with canonical axes.
    name: Optional op name.

  Returns:
    The prediction tensors in canonical prediction format.
  """
  with tf.name_scope(name, 'model', [input_lt]) as scope:
    input_lt = tensorcheck.bounds(0.0, 1.0, input_lt)

    for a in pp.target_axes:
      logging.info('a: %r', a)
      assert a['z'].labels is not None
      assert a['channel'].labels is not None

    input_lt = lt.transpose(input_lt, util.CANONICAL_AXIS_ORDER)
    input_lt = lt.reshape(input_lt, ['z', 'channel', 'mask'], ['depth'])
    input_op = input_lt.tensor

    core_model_op = core_model(is_train=is_train, input_op=input_op, name=name)

    # Define output
    output_lts = []
    for i, axes in enumerate(pp.target_axes):
      z_lts = []
      for z in axes['z'].labels:
        channel_lts = []
        for channel in axes['channel'].labels:
          op_name = 'head/%s/%s' % (z, channel)

          output_op = add_head(
              num_classes=pp.num_classes,
              is_train=is_train,
              input_op=core_model_op,
              name=op_name)

          output_lt = lt.LabeledTensor(output_op, [
              input_lt.axes['batch'], 'row', 'column', ('class', pp.num_classes)
          ])

          channel_lts.append(
              lt.expand_dims(output_lt, [
                  'batch', 'row', 'column', ('z', z),
                  ('channel', channel), 'class'
              ]))

        z_lts.append(lt.concat(channel_lts, 'channel'))
      output_lts.append(lt.concat(z_lts, 'z', name=scope + str(i)))

    return output_lts


def get_input_target_and_predicted(
    gitapp: GetInputTargetAndPredictedParameters,
    name: str = None,
) -> Tuple[np.ndarray, lt.LabeledTensor, lt.LabeledTensor, lt.LabeledTensor,
           lt.LabeledTensor]:
  """Read and preprocess Seeing More data, and run it through the model.

  Args:
    gitapp: GetInputTargetAndPredictedParameters.
    name: Optional op name.

  Returns:
    An array containing patch center locations.
    The patches are extracted with padding, so that the stitched model outputs
    will form an image the same size as the input.

    A tensor in the canonical format with model inputs, possibly jittered and
    corrupted for data enrichment.

    A tensor with model targets in the canonical format.

    A tensor with predicted inputs in the canonical prediction format.

    A tensor with predicted targets in the canonical prediction format.
  """
  logging.info('Getting model input, target, and predicted.')
  with tf.name_scope(name, 'get_input_target_and_predicted', []) as scope:
    # Do all preprocessing on the CPU, to avoid contending with the GPU if
    # it's evaluating the model.
    # with tf.device('/cpu:0'):
    patch_centers, input_lt, target_lt = provide_preprocessed_data(
        gitapp.dp, gitapp.ap, gitapp.extract_patch_size, gitapp.stride)

    if gitapp.bp is not None:
      if gitapp.shuffle:
        input_lt, target_lt = lt.shuffle_batch(
            [input_lt, target_lt],
            name='preprocess',
            batch_size=gitapp.bp.size,
            num_threads=gitapp.bp.num_threads,
            capacity=gitapp.bp.capacity,
            enqueue_many=True,
            min_after_dequeue=gitapp.bp.capacity // 2)
      else:
        input_lt, target_lt = util.entry_point_batch(
            [input_lt, target_lt],
            bp=gitapp.bp,
            enqueue_many=True,
            entry_point_names=['input_patch', 'target_patch'],
            name='preprocess')

    pp = model_util.PredictionParameters(
        [(input_lt.axes['z'].labels, input_lt.axes['channel'].labels),
         (target_lt.axes['z'].labels, target_lt.axes['channel'].labels)],
        gitapp.num_classes)
    # pylint: disable=unbalanced-tuple-unpacking
    predict_input_lt, predict_target_lt = model(
        gitapp.core_model, gitapp.add_head, pp, gitapp.is_train, input_lt)
    # pylint: enable=unbalanced-tuple-unpacking
    # Ensure the model output size is as we expect.
    assert len(predict_input_lt.axes[
        'row']) == gitapp.stitch_patch_size, predict_input_lt
    assert len(predict_input_lt.axes[
        'column']) == gitapp.stitch_patch_size, predict_input_lt
    assert len(predict_target_lt.axes[
        'row']) == gitapp.stitch_patch_size, predict_target_lt
    assert len(predict_target_lt.axes[
        'column']) == gitapp.stitch_patch_size, predict_target_lt

    input_lt = lt.identity(input_lt, name=scope + 'input')
    target_lt = lt.identity(target_lt, name=scope + 'target')
    predict_input_lt = lt.identity(
        predict_input_lt, name=scope + 'predict_input')
    predict_target_lt = lt.identity(
        predict_target_lt, name=scope + 'predict_target')

    return (patch_centers, input_lt, target_lt, predict_input_lt,
            predict_target_lt)


def add_loss(
    loss: Callable,
    target_lt: lt.LabeledTensor,
    predicted_lt: lt.LabeledTensor,
    name: str = None,
) -> lt.LabeledTensor:
  """Add a loss.

  Args:
    loss: Loss function to use.
      Arguments should be (target, mask, prediction, name).
    target_lt: The target values in the canonical format.
    predicted_lt: The predicted values in the canonical prediction format.
    name: Optional op name.

  Returns:
    A scalar tensor representing the weighted cross-entropy loss.
  """
  with tf.name_scope(name, 'loss', [target_lt, predicted_lt]) as scope:
    target_lt = lt.transpose(target_lt, util.CANONICAL_AXIS_ORDER)
    predicted_lt = lt.transpose(predicted_lt,
                                util.CANONICAL_PREDICTION_AXIS_ORDER)

    predicted_size = len(predicted_lt.axes['row'])
    assert predicted_size == len(predicted_lt.axes['column'])
    target_lt = util.crop_center(predicted_size, target_lt)

    signal_lt = lt.select(target_lt, {'mask': False})
    mask_lt = lt.select(target_lt, {'mask': True})

    signal_lt = lt.reshape(signal_lt, util.CANONICAL_AXIS_ORDER[:-1], ['batch'])
    mask_lt = lt.reshape(mask_lt, util.CANONICAL_AXIS_ORDER[:-1], ['batch'])
    predicted_lt = lt.reshape(
        predicted_lt, util.CANONICAL_PREDICTION_AXIS_ORDER[:-1], ['batch'])

    assert list(signal_lt.axes.keys()) == ['batch']
    assert list(mask_lt.axes.keys()) == ['batch']
    assert list(predicted_lt.axes.keys()) == ['batch', 'class']

    signal_lt = tensorcheck.bounds(0.0, 1.0, signal_lt)
    mask_lt = tensorcheck.bounds(0.0, 1.0, mask_lt)

    loss_lt = loss(signal_lt, mask_lt, predicted_lt)

    return lt.identity(loss_lt, name=scope)


def itemize_losses(
    loss: Callable,
    target_lt: lt.LabeledTensor,
    predict_lt: lt.LabeledTensor,
    name: str = None,
) -> Dict[str, lt.LabeledTensor]:
  """Create itemized losses for each prediction task.

  Creates named losses for each prediction task.

  Args:
    loss: Loss function to use.
      Arguments should be (target, mask, prediction, name).
    target_lt: Tensor with ground truth values, in canonical format.
    predict_lt: Tensor with predicted logits, in canonical prediction format.
    name: Optional op name.

  Returns:
    A dictionary mapping loss names to loss tensors.
  """
  with tf.name_scope(name, 'itemize_losses', [target_lt, predict_lt]) as scope:
    loss_lts = {}
    axes = target_lt.axes
    for z in axes['z'].labels:
      for channel in axes['channel'].labels:
        target_selection_lt = lt.select(target_lt, {
            'z': util.slice_1(z),
            'channel': util.slice_1(channel)
        })
        predict_selection_lt = lt.select(predict_lt, {
            'z': util.slice_1(z),
            'channel': util.slice_1(channel)
        })
        tag = '%s/%s' % (z, channel)
        loss_lt = add_loss(
            loss, target_selection_lt, predict_selection_lt, name=scope + tag)
        tf.summary.scalar(name=os.path.join('loss', tag), tensor=loss_lt.tensor)
        tf.summary.histogram(
            name=os.path.join('loss', tag, 'target'),
            values=target_selection_lt.tensor)
        tf.summary.histogram(
            name=os.path.join('loss', tag, 'predict'),
            values=predict_selection_lt.tensor)
        loss_lts[tag] = loss_lt

    return loss_lts


def setup_losses(
    gitapp: GetInputTargetAndPredictedParameters,
    name: str = None,
) -> Tuple[Dict[str, lt.LabeledTensor], Dict[str, lt.LabeledTensor]]:
  """Creates cross entropy losses.

  Args:
    gitapp: GetInputTargetAndPredictedParameters.
    name: Optional op name.

  Returns:
    A dictionary of tensors with the input reconstruction losses.

    A dictionary of tensors with the target prediction losses.
  """
  logging.info('Setting up losses')
  with tf.name_scope(name, 'setup_losses', []) as scope:
    (_, input_lt, target_lt, predict_input_lt,
     predict_target_lt) = get_input_target_and_predicted(gitapp)

    predicted_size = len(predict_input_lt.axes['row'])
    visualize.summarize_image(
        visualize.error_panel(
            util.crop_center(predicted_size, input_lt),
            visualize.to_softmax(predict_input_lt),
            name=scope + 'input_patch_error_panel'))
    visualize.summarize_image(
        visualize.error_panel(
            util.crop_center(predicted_size, target_lt),
            visualize.to_softmax(predict_target_lt),
            name=scope + 'target_patch_error_panel'))

    def mean(lts: Dict[str, lt.LabeledTensor]) -> tf.Tensor:
      sum_op = tf.add_n([t.tensor for t in lts.values()])
      return sum_op / float(len(lts))

    tag = 'input'
    input_loss_lts = itemize_losses(
        gitapp.loss, input_lt, predict_input_lt, name=scope + tag)
    tf.summary.scalar(name='loss/' + tag, tensor=mean(input_loss_lts))

    tag = 'target'
    target_loss_lts = itemize_losses(
        gitapp.loss, target_lt, predict_target_lt, name=scope + tag)
    tf.summary.scalar(name='loss/' + tag, tensor=mean(target_loss_lts))

    variables = tf.global_variables()
    for v in variables:
      tf.summary.histogram(name='variable/' + v.name, values=v)

    return input_loss_lts, target_loss_lts


def setup_stitch(
    gitapp: GetInputTargetAndPredictedParameters,
    name=None,
) -> Dict[str, lt.LabeledTensor]:
  """Creates diagnostic images.

  All diagnostic images are registered as summaries.

  Args:
    gitapp: GetInputTargetAndPredictedParameters.
    name: Optional op name.

  Returns:
    A mapping where the keys are names of summary images and the values
    are image tensors.
  """
  logging.info('Setting up stitch')
  with tf.name_scope(name, 'setup_stitch', []) as scope:
    (patch_centers, input_lt, target_lt, predict_input_lt,
     predict_target_lt) = get_input_target_and_predicted(gitapp)

    predicted_size = len(predict_input_lt.axes['row'])
    assert predicted_size == len(predict_input_lt.axes['column'])
    input_lt = util.crop_center(predicted_size, input_lt)
    target_lt = util.crop_center(predicted_size, target_lt)

    # For now, we're not handling overlap or missing data.
    assert gitapp.stride == predicted_size

    if gitapp.bp is not None:
      # Rebatch so a single tensor is all the patches in a single image.
      [input_lt, target_lt, predict_input_lt,
       predict_target_lt] = util.entry_point_batch(
           [input_lt, target_lt, predict_input_lt, predict_target_lt],
           bp=util.BatchParameters(
               size=len(patch_centers), num_threads=1, capacity=1),
           enqueue_many=True,
           entry_point_names=[
               'input_stitch', 'target_stitch', 'predict_input_stitch',
               'predict_target_stitch'
           ],
           name='stitch')

    rc = lt.ReshapeCoder(util.CANONICAL_AXIS_ORDER[3:], ['channel'])
    input_lt = rc.decode(
        ops.patches_to_image(patch_centers, rc.encode(input_lt)))

    rc = lt.ReshapeCoder(util.CANONICAL_AXIS_ORDER[3:], ['channel'])
    target_lt = rc.decode(
        ops.patches_to_image(patch_centers, rc.encode(target_lt)))

    rc = lt.ReshapeCoder(util.CANONICAL_PREDICTION_AXIS_ORDER[3:], ['channel'])
    predict_input_lt = rc.decode(
        ops.patches_to_image(patch_centers, rc.encode(predict_input_lt)))

    rc = lt.ReshapeCoder(util.CANONICAL_PREDICTION_AXIS_ORDER[3:], ['channel'])
    predict_target_lt = rc.decode(
        ops.patches_to_image(patch_centers, rc.encode(predict_target_lt)))

    def get_statistics(t: lt.LabeledTensor) -> lt.LabeledTensor:
      t = visualize.to_softmax(t)
      rc = lt.ReshapeCoder(list(t.axes.keys())[:-1], ['batch'])
      return rc.decode(ops.distribution_statistics(rc.encode(t)))

    # C++ entry points .
    with tf.name_scope(''):
      input_lt = lt.identity(input_lt, name='entry_point_stitched_input')
      target_lt = lt.identity(target_lt, name='entry_point_stitched_target')
      # The nodes are used purely to export data to C++.
      lt.identity(
          get_statistics(predict_input_lt),
          name='entry_point_stitched_predicted_input')
      lt.identity(
          get_statistics(predict_target_lt),
          name='entry_point_stitched_predicted_target')

    predict_input_lt = visualize.to_softmax(predict_input_lt)
    predict_target_lt = visualize.to_softmax(predict_target_lt)

    input_summary_lt = visualize.error_panel(input_lt, predict_input_lt)
    target_summary_lt = visualize.error_panel(target_lt, predict_target_lt)

    if gitapp.bp is not None:
      input_summary_lt, target_summary_lt = lt.batch(
          [input_summary_lt, target_summary_lt],
          # We'll see 3 images in the visualizer.
          batch_size=3,
          enqueue_many=True,
          num_threads=1,
          capacity=1,
          name='group')

    input_summary_lt = lt.identity(
        input_summary_lt, name=scope + 'input_error_panel')
    target_summary_lt = lt.identity(
        target_summary_lt, name=scope + 'target_error_panel')

    visualize_op_dict = {}
    visualize_op_dict['input'] = input_lt
    visualize_op_dict['predict_input'] = predict_input_lt
    visualize_op_dict['target'] = target_lt
    visualize_op_dict['predict_target'] = predict_target_lt

    def summarize(tag, labeled_tensor):
      visualize.summarize_image(labeled_tensor, name=scope + 'summarize/' + tag)
      visualize_op_dict[tag] = labeled_tensor

    summarize('input_error_panel', input_summary_lt)
    summarize('target_error_panel', target_summary_lt)

    return visualize_op_dict
