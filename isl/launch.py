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

# pylint: disable=line-too-long
r"""Binary for training and evaluating a model."""
# pylint: enable=line-too-long
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import math
import os
import time

import numpy as np
import tensorflow as tf
from typing import Dict, List, Tuple

# pylint: disable=g-bad-import-order
from isl import augment
from isl import controller
from isl import data_provider
from isl import infer
from isl import util
from isl.models import concordance
from isl.models import model_util

slim = tf.contrib.slim
metrics = tf.contrib.metrics
app = tf.app
logging = tf.logging
flags = tf.flags
gfile = tf.gfile
lt = tf.contrib.labeled_tensor

MODE_TRAIN = 'TRAIN'
MODE_EVAL_TRAIN = 'EVAL_TRAIN'
MODE_EVAL_EVAL = 'EVAL_EVAL'
MODE_EXPORT = 'EXPORT'
flags.DEFINE_string('mode', MODE_TRAIN, 'What this binary will do.')

METRIC_LOSS = 'LOSS'
METRIC_JITTER_STITCH = 'JITTER_STITCH'
METRIC_STITCH = 'STITCH'
METRIC_INFER_FULL = 'INFER_FULL'
flags.DEFINE_string('metric', METRIC_LOSS, 'What this binary will display.')

flags.DEFINE_string('master', 'local',
                    'BNS name of the TensorFlow master to use.')
flags.DEFINE_string('base_directory', '/tmp/minception/',
                    'Directory where model checkpoints are written.')
flags.DEFINE_string('export_directory', '/tmp/minception_export/',
                    'Directory where exported model is written.')
flags.DEFINE_integer(
    'save_summaries_secs', 180,
    'The frequency with which summaries are saved, in seconds.')
flags.DEFINE_integer('save_interval_secs', 180,
                     'The frequency with which the model is saved, in seconds.')
flags.DEFINE_integer('eval_interval_secs', 15,
                     'The frequency, in seconds, with which evaluation is run.')
flags.DEFINE_integer('eval_delay_secs', 0,
                     'The time to wait before starting evaluations.')
flags.DEFINE_integer(
    'metric_num_examples', 1 << 10,
    'The number of examples to use when computing tf.slim metrics.')
flags.DEFINE_integer(
    'ps_tasks', 0,
    'The number of parameter servers. If the value is 0, then the parameters '
    'are handled locally by the worker.')
flags.DEFINE_integer(
    'task', 0,
    'The Task ID. This value is used when training with multiple workers to '
    'identify each worker.')
flags.DEFINE_string(
    'restore_directory', '',
    'If provided, the directory from which to restore a model checkpoint for '
    'training or exporting.')

OPTIMIZER_MOMENTUM = 'MOMENTUM'
OPTIMIZER_ADAGRAD = 'ADAGRAD'
OPTIMIZER_ADAM = 'ADAM'
flags.DEFINE_string('optimizer', 'ADAM', 'The train optimizer.')

flags.DEFINE_float('learning_rate', 1e-4, 'The learning rate.')

flags.DEFINE_integer(
    'learning_decay_steps', 1 << 12,
    'The learning decay steps, used by the MOMENTUM optimizer.')

flags.DEFINE_bool(
    'read_pngs', False,
    'Whether to read the input images from a provided folder rather than '
    'from a RecordIO or SSTable.')

# Parameters for when read_pngs is True.
flags.DEFINE_string('dataset_train_directory', None,
                    'If read_pngs, the directory containing the train dataset.')
flags.DEFINE_string(
    'dataset_eval_directory', None,
    'If read_pngs, the directory containing the evaluation dataset.')

# Parameters for when read_pngs is False.
flags.DEFINE_string(
    'dataset_pattern', None,
    'If not read_pngs, format string giving the dataset location. '
    'It will be subdivided into train and eval sets.')
flags.DEFINE_integer('dataset_num_shards', 1024,
                     'If not read_pngs, the number of shards in the dataset.')
flags.DEFINE_bool(
    'is_recordio', True,
    'If not read_pngs, whether the dataset is stored as a RecordIO, '
    'else an SSTable.')
flags.DEFINE_integer(
    'data_batch_size', 4,
    'If not read_pngs, batch size for first part of preprocessing.')
flags.DEFINE_integer(
    'data_batch_num_threads', 4,
    'If not read_pngs, number of threads loading data from disk.')
flags.DEFINE_integer(
    'data_batch_capacity', 8,
    'If not read_pngs, batch capacity for threads loading data from disk.')

flags.DEFINE_integer('loss_crop_size', 520, 'Image crop size for training.')
flags.DEFINE_integer('loss_patch_stride', 256, '')
flags.DEFINE_integer('stitch_crop_size', 500, 'Image crop size for stitching.')
flags.DEFINE_integer(
    'infer_size', 16,
    'The number of inferences to do in parallel in each row x column dimension.'
    ' For example, a size of 16 will do 16 x 16 = 256 inferences in parallel.')
flags.DEFINE_bool('infer_continuously', False,
                  'Whether to run inference in a while loop.')
flags.DEFINE_string('infer_channel_whitelist', None,
                    'If provided, the channels to whitelist for inference.')
flags.DEFINE_bool('infer_simplify_error_panels', True,
                  'Whether to simplify the error panels.')

flags.DEFINE_float('augment_offset_std', 0.0,
                   'Augmentation noise corruption parameter.')
flags.DEFINE_float('augment_multiplier_std', 0.0,
                   'Augmentation noise corruption parameter.')
flags.DEFINE_float('augment_noise_std', 0.0,
                   'Augmentation noise corruption parameter.')

flags.DEFINE_integer('preprocess_batch_size', 16, 'Batch size for the model.')
flags.DEFINE_integer(
    'preprocess_shuffle_batch_num_threads', 16,
    'Number of threads doing the second half of preprocessing during training.')
flags.DEFINE_integer('preprocess_batch_capacity', 64,
                     'Batch capacity for second half of preprocessing.')

flags.DEFINE_bool(
    'train_on_full_dataset', False,
    'If true, train on the full dataset, not the subset used for training in '
    'the training / evaluation split. Useful for getting the last bit of '
    'performance out of a model we trust.')

LOSS_CROSS_ENTROPY = 'CROSS_ENTROPY'
LOSS_RANKED_PROBABILITY_SCORE = 'RANKED_PROBABILITY_SCORE'
flags.DEFINE_string('loss', LOSS_CROSS_ENTROPY, 'The loss to use.')

MODEL_CONCORDANCE = 'CONCORDANCE'
flags.DEFINE_string('model', MODEL_CONCORDANCE, 'The network model to use.')

flags.DEFINE_integer('base_depth', 400, 'Model parameter.')

flags.DEFINE_bool(
    'restore_logits', True,
    'Whether to restore the heads when resuming training. Set to False if you '
    'want to add or remove heads but restore the rest of the network.')
flags.DEFINE_bool(
    'restore_inputs', True,
    'Whether to restore the input layers when resuming training. Set to False '
    'if you are restoring from another model checkpoint where the input '
    'dimension does not match the input dimension of this network.  For '
    'instance, the DeepLab pretrained models only have three channels for RGB.')

flags.DEFINE_integer('num_z_values', 26,
                     'Number of z depths to use from input.')

FLAGS = flags.FLAGS


# TODO(ericmc): Consider simplifying this using np.linspace.
def get_z_values() -> List[float]:
  """Gets the z-values the model will take as input."""
  values = np.linspace(0.0, 1.0, FLAGS.num_z_values)
  values = [round(v, 4) for v in values]
  logging.info('z_values: %r', values)
  return values


INPUT_CHANNEL_VALUES = [
    'BRIGHTFIELD',
    'PHASE_CONTRAST',
    'DIC',
]
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

# The size of the extracted input patches.
CONCORDANCE_EXTRACT_PATCH_SIZE = 250
# The size of the model output.
CONCORDANCE_STITCH_PATCH_SIZE = 8
# The stride to use when stitching.
# It is almost always equal to STITCH_PATCH_SIZE except when debugging.
CONCORDANCE_STITCH_STRIDE = 8

# The number of classes into which to bin pixel values.
NUM_CLASSES = 256


def data_parameters() -> data_provider.DataParameters:
  """Creates the DataParameters."""
  if FLAGS.read_pngs:
    if FLAGS.mode == MODE_TRAIN or FLAGS.mode == MODE_EVAL_TRAIN:
      directory = FLAGS.dataset_train_directory
    else:
      directory = FLAGS.dataset_eval_directory

    if FLAGS.metric == METRIC_LOSS:
      crop_size = FLAGS.loss_crop_size
    else:
      crop_size = FLAGS.stitch_crop_size

    io_parameters = data_provider.ReadPNGsParameters(directory, None, None,
                                                     crop_size)
  else:
    # Use an eighth of the dataset for validation.
    if FLAGS.mode == MODE_TRAIN or FLAGS.mode == MODE_EVAL_TRAIN:
      dataset = [
          FLAGS.dataset_pattern % i
          for i in range(FLAGS.dataset_num_shards)
          if (i % 8 != 0) or FLAGS.train_on_full_dataset
      ]
    else:
      dataset = [
          FLAGS.dataset_pattern % i
          for i in range(FLAGS.dataset_num_shards)
          if i % 8 == 0
      ]
    if FLAGS.metric == METRIC_LOSS:
      crop_size = FLAGS.loss_crop_size
    else:
      crop_size = FLAGS.stitch_crop_size

    if FLAGS.model == MODEL_CONCORDANCE:
      extract_patch_size = CONCORDANCE_EXTRACT_PATCH_SIZE
      stitch_patch_size = CONCORDANCE_STITCH_PATCH_SIZE
    else:
      raise NotImplementedError('Unsupported model: %s' % FLAGS.model)

    if FLAGS.mode == MODE_EXPORT:
      # Any padding will be done by the C++ caller.
      pad_width = 0
    else:
      pad_width = (extract_patch_size - stitch_patch_size) // 2

    io_parameters = data_provider.ReadTableParameters(
        dataset,
        FLAGS.is_recordio,
        util.BatchParameters(FLAGS.data_batch_size,
                             FLAGS.data_batch_num_threads,
                             FLAGS.data_batch_capacity),
        # Do non-deterministic data fetching, to increase the variety of what we
        # see in the visualizer.
        False,
        pad_width,
        crop_size)

  z_values = get_z_values()
  return data_provider.DataParameters(io_parameters, z_values,
                                      INPUT_CHANNEL_VALUES, TARGET_Z_VALUES,
                                      TARGET_CHANNEL_VALUES)


def parameters() -> controller.GetInputTargetAndPredictedParameters:
  """Creates the network parameters for the given inputs and flags.

  Returns:
    A GetInputTargetAndPredictedParameters containing network parameters for the
    given mode, metric, and other flags.
  """
  if FLAGS.metric == METRIC_LOSS:
    stride = FLAGS.loss_patch_stride
    shuffle = True
  else:
    if FLAGS.model == MODEL_CONCORDANCE:
      stride = CONCORDANCE_STITCH_STRIDE
    else:
      raise NotImplementedError('Unsupported model: %s' % FLAGS.model)
    # Shuffling breaks stitching.
    shuffle = False

  if FLAGS.mode == MODE_TRAIN:
    is_train = True
  else:
    is_train = False

  if FLAGS.model == MODEL_CONCORDANCE:
    core_model = functools.partial(concordance.core, FLAGS.base_depth)
    add_head = functools.partial(model_util.add_head, is_residual_conv=True)
    extract_patch_size = CONCORDANCE_EXTRACT_PATCH_SIZE
    stitch_patch_size = CONCORDANCE_STITCH_PATCH_SIZE
  else:
    raise NotImplementedError('Unsupported model: %s' % FLAGS.model)

  dp = data_parameters()

  if shuffle:
    preprocess_num_threads = FLAGS.preprocess_shuffle_batch_num_threads
  else:
    # Thread racing is an additional source of shuffling, so we can only
    # use 1 thread per queue.
    preprocess_num_threads = 1
  if is_train or FLAGS.metric == METRIC_JITTER_STITCH:
    ap = augment.AugmentParameters(FLAGS.augment_offset_std,
                                   FLAGS.augment_multiplier_std,
                                   FLAGS.augment_noise_std)
  else:
    ap = None

  if FLAGS.metric == METRIC_INFER_FULL:
    bp = None
  else:
    bp = util.BatchParameters(FLAGS.preprocess_batch_size,
                              preprocess_num_threads,
                              FLAGS.preprocess_batch_capacity)

  if FLAGS.loss == LOSS_CROSS_ENTROPY:
    loss = util.softmax_cross_entropy
  elif FLAGS.loss == LOSS_RANKED_PROBABILITY_SCORE:
    loss = util.ranked_probability_score
  else:
    logging.fatal('Invalid loss: %s', FLAGS.loss)

  return controller.GetInputTargetAndPredictedParameters(
      dp, ap, extract_patch_size, stride, stitch_patch_size, bp, core_model,
      add_head, shuffle, NUM_CLASSES, loss, is_train)


def train_directory() -> str:
  """The directory where the training data is written."""
  return os.path.join(FLAGS.base_directory, 'train')


def output_directory() -> str:
  """The output directory for the current invocation of this binary."""
  if FLAGS.mode == MODE_TRAIN:
    return train_directory()
  else:
    if FLAGS.mode == MODE_EVAL_TRAIN:
      prefix = 'eval_train_'
    else:
      prefix = 'eval_eval_'

    if FLAGS.metric == METRIC_INFER_FULL:
      suffix = 'infer'
    elif FLAGS.metric == METRIC_LOSS:
      suffix = 'loss_' + FLAGS.loss
    elif FLAGS.metric == METRIC_JITTER_STITCH:
      suffix = 'jitter_stitch'
    else:
      suffix = 'stitch'

    return os.path.join(FLAGS.base_directory, prefix + suffix)


def total_loss(
    gitapp: controller.GetInputTargetAndPredictedParameters,
) -> Tuple[tf.Tensor, Dict[str, lt.LabeledTensor], Dict[str, lt.LabeledTensor]]:
  """Get the total weighted training loss."""
  input_loss_lts, target_loss_lts = controller.setup_losses(gitapp)

  def mean(lts: Dict[str, lt.LabeledTensor]) -> tf.Tensor:
    sum_op = tf.add_n([t.tensor for t in lts.values()])
    return sum_op / float(len(lts))

  # Give the input loss the same weight as the target loss.
  input_weight = 0.5
  total_loss_op = input_weight * mean(input_loss_lts) + (
      1 - input_weight) * mean(target_loss_lts)
  tf.summary.scalar('total_loss', total_loss_op)

  return total_loss_op, input_loss_lts, target_loss_lts


def log_entry_points(g: tf.Graph):
  logging.info('Entry points: %s',
               [o.name for o in g.get_operations() if 'entry_point' in o.name])


def train(gitapp: controller.GetInputTargetAndPredictedParameters):
  """Train a model."""
  g = tf.Graph()
  with g.as_default():
    total_loss_op, _, _ = total_loss(gitapp)

    if FLAGS.optimizer == OPTIMIZER_MOMENTUM:
      # TODO(ericmc): We may want to do weight decay with the other
      # optimizers, too.
      learning_rate = tf.train.exponential_decay(
          FLAGS.learning_rate,
          slim.variables.get_global_step(),
          FLAGS.learning_decay_steps,
          0.999,
          staircase=False)
      tf.summary.scalar('learning_rate', learning_rate)

      optimizer = tf.train.MomentumOptimizer(learning_rate, 0.875)
    elif FLAGS.optimizer == OPTIMIZER_ADAGRAD:
      optimizer = tf.train.AdagradOptimizer(FLAGS.learning_rate)
    elif FLAGS.optimizer == OPTIMIZER_ADAM:
      optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    else:
      raise NotImplementedError('Unsupported optimizer: %s' % FLAGS.optimizer)

    # Set up training.
    train_op = slim.learning.create_train_op(
        total_loss_op, optimizer, summarize_gradients=True)

    if FLAGS.restore_directory:
      init_fn = util.restore_model(FLAGS.restore_directory,
                                   FLAGS.restore_logits)

    else:
      logging.info('Training a new model.')
      init_fn = None

    total_variable_size, _ = slim.model_analyzer.analyze_vars(
        slim.get_variables(), print_info=True)
    logging.info('Total number of variables: %d', total_variable_size)

    log_entry_points(g)

    slim.learning.train(
        train_op=train_op,
        logdir=output_directory(),
        master=FLAGS.master,
        is_chief=FLAGS.task == 0,
        number_of_steps=None,
        save_summaries_secs=FLAGS.save_summaries_secs,
        save_interval_secs=FLAGS.save_interval_secs,
        init_fn=init_fn,
        saver=tf.train.Saver(keep_checkpoint_every_n_hours=2.0))


def eval_loss(gitapp: controller.GetInputTargetAndPredictedParameters):
  g = tf.Graph()
  with g.as_default():
    total_loss_op, input_loss_lts, target_loss_lts = total_loss(gitapp)

    metric_names = ['total_loss']
    metric_values = [total_loss_op]
    for name, loss_lt in dict(input_loss_lts, **target_loss_lts).items():
      metric_names.append(name)
      metric_values.append(loss_lt.tensor)
    metric_names = ['metric/' + n for n in metric_names]
    metric_values = [metrics.streaming_mean(v) for v in metric_values]

    names_to_values, names_to_updates = metrics.aggregate_metric_map(
        dict(zip(metric_names, metric_values)))

    for name, value in names_to_values.iteritems():
      slim.summaries.add_scalar_summary(value, name, print_summary=True)

    log_entry_points(g)

    num_batches = FLAGS.metric_num_examples // gitapp.bp.size

    slim.evaluation.evaluation_loop(
        master=FLAGS.master,
        checkpoint_dir=train_directory(),
        logdir=output_directory(),
        num_evals=num_batches,
        eval_op=names_to_updates.values(),
        eval_interval_secs=FLAGS.eval_interval_secs)


def eval_stitch(gitapp: controller.GetInputTargetAndPredictedParameters):
  g = tf.Graph()
  with g.as_default():
    controller.setup_stitch(gitapp)

    summary_ops = tf.get_collection(tf.GraphKeys.SUMMARIES)
    input_summary_op = next(
        x for x in summary_ops if 'input_error_panel' in x.name)
    target_summary_op = next(
        x for x in summary_ops if 'target_error_panel' in x.name)

    log_entry_points(g)

    slim.evaluation.evaluation_loop(
        master=FLAGS.master,
        num_evals=0,
        checkpoint_dir=train_directory(),
        logdir=output_directory(),
        # Merge the summaries to keep the graph state in sync.
        summary_op=tf.summary.merge([input_summary_op, target_summary_op]),
        eval_interval_secs=FLAGS.eval_interval_secs)


def export(gitapp: controller.GetInputTargetAndPredictedParameters):
  g = tf.Graph()
  with g.as_default():
    assert FLAGS.metric == METRIC_STITCH

    controller.setup_stitch(gitapp)

    log_entry_points(g)

    signature_map = dict(
        [(o.name, o) for o in g.get_operations() if 'entry_point' in o.name])

    logging.info('Exporting checkpoint at %s to %s', FLAGS.restore_directory,
                 FLAGS.export_directory)
    slim.export_for_serving(
        g,
        checkpoint_dir=FLAGS.restore_directory,
        export_dir=FLAGS.export_directory,
        generic_signature_tensor_map=signature_map)


def infer_single_image(gitapp: controller.GetInputTargetAndPredictedParameters):
  """Predicts the labels for a single image."""
  if not gfile.Exists(output_directory()):
    gfile.MakeDirs(output_directory())

  if FLAGS.infer_channel_whitelist is not None:
    infer_channel_whitelist = FLAGS.infer_channel_whitelist.split(',')
  else:
    infer_channel_whitelist = None

  while True:
    infer.infer(
        gitapp=gitapp,
        restore_directory=FLAGS.restore_directory or train_directory(),
        output_directory=output_directory(),
        extract_patch_size=CONCORDANCE_EXTRACT_PATCH_SIZE,
        stitch_stride=CONCORDANCE_STITCH_STRIDE,
        infer_size=FLAGS.infer_size,
        channel_whitelist=infer_channel_whitelist,
        simplify_error_panels=FLAGS.infer_simplify_error_panels,
    )
    if not FLAGS.infer_continuously:
      break


def main(_):
  logging.set_verbosity("INFO")
  if FLAGS.mode == MODE_TRAIN:
    assert FLAGS.metric == METRIC_LOSS

  if FLAGS.task == 0 and not gfile.Exists(FLAGS.base_directory):
    gfile.MakeDirs(FLAGS.base_directory)

  gitapp = parameters()
  if FLAGS.metric == METRIC_INFER_FULL:
    infer_single_image(gitapp)
  elif FLAGS.mode == MODE_TRAIN:
    train(gitapp)
  elif FLAGS.mode == MODE_EXPORT:
    export(gitapp)
  elif FLAGS.metric == METRIC_LOSS:
    logging.info('Sleeping %d seconds before beginning evaluation',
                 FLAGS.eval_delay_secs)
    time.sleep(FLAGS.eval_delay_secs)
    eval_loss(gitapp)
  elif FLAGS.metric == METRIC_JITTER_STITCH or FLAGS.metric == METRIC_STITCH:
    logging.info('Sleeping %d seconds before beginning evaluation',
                 FLAGS.eval_delay_secs)
    time.sleep(FLAGS.eval_delay_secs)
    eval_stitch(gitapp)
  else:
    raise NotImplementedError


if __name__ == '__main__':
  app.run()
