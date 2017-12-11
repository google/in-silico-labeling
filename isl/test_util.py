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
"""Test util for Seeing More Tensorflow libraries."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import os

import numpy as np
import tensorflow as tf
from typing import List, Tuple, Union

# pylint: disable=g-bad-import-order
from isl import data_provider
from isl import util

flags = tf.flags
gfile = tf.gfile
logging = tf.logging
lt = tf.contrib.labeled_tensor

FLAGS = flags.FLAGS


class Base(tf.test.TestCase):
  """Convenience base class for writing tests."""

  def setUp(self):
    super(Base, self).setUp()

    self.input_z_values = [round(v, 4) for v in np.linspace(0.0, 1.0, 13)]
    self.input_channel_values = ['BRIGHTFIELD', 'PHASE_CONTRAST', 'DIC']
    self.target_z_values = ['MAXPROJECT']
    self.target_channel_values = [
        'DAPI_CONFOCAL',
        'DAPI_WIDEFIELD',
        'CELLMASK_CONFOCAL',
        'TUJ1_WIDEFIELD',
        'NFH_CONFOCAL',
        'MAP2_CONFOCAL',
        'ISLET_WIDEFIELD',
        'DEAD_CONFOCAL',
    ]

    rtp = data_provider.ReadTableParameters([self.recordio_path()], True,
                                            util.BatchParameters(2, 1, 2), True,
                                            0, 256)
    dp = data_provider.DataParameters(
        rtp, self.input_z_values, self.input_channel_values,
        self.target_z_values, self.target_channel_values)
    # pylint: disable=line-too-long
    self.input_lt, self.target_lt = data_provider.cropped_input_and_target(dp)
    # pylint: enable=line-too-long

    minception_rtp = data_provider.ReadTableParameters([self.recordio_path()],
                                                       True,
                                                       util.BatchParameters(
                                                           2, 1,
                                                           2), True, 51, 110)
    minception_dp = data_provider.DataParameters(
        minception_rtp, self.input_z_values, self.input_channel_values,
        self.target_z_values, self.target_channel_values)
    # pylint: disable=line-too-long
    self.minception_input_lt, self.minception_target_lt = data_provider.cropped_input_and_target(
        minception_dp)
    # pylint: enable=line-too-long

  def data_path(self, filename: str) -> str:
    return os.path.join(os.environ['TEST_SRCDIR'], 'isl/testdata', filename)

  def load_tensorflow_image(self, channel_label: str,
                            image_name: str) -> lt.LabeledTensor:
    # All images will be cropped to this size.
    crop_size = 1024

    filename_op = tf.train.string_input_producer([self.data_path(image_name)])
    wfr = tf.WholeFileReader()
    _, encoded_png_op = wfr.read(filename_op)
    image_op = tf.image.decode_png(
        tf.reshape(encoded_png_op, shape=[]), channels=1, dtype=tf.uint16)
    image_op = image_op[:crop_size, :crop_size, :]
    image_op = tf.to_float(image_op) / np.iinfo(np.uint16).max
    image_op = tf.reshape(image_op, [1, 1024, 1024, 1])

    return lt.LabeledTensor(
        image_op, ['batch', 'row', 'column', ('channel', [channel_label])])

  def load_dic(self) -> lt.LabeledTensor:
    return self.load_tensorflow_image('DIC', 'dic.png')

  def load_dapi(self) -> lt.LabeledTensor:
    return self.load_tensorflow_image('DAPI_WIDEFIELD', 'dapi_widefield.png')

  def load_tuj1(self) -> lt.LabeledTensor:
    return self.load_tensorflow_image('TUJ1_WIDEFIELD', 'tuj1_widefield.png')

  def load_map2(self) -> lt.LabeledTensor:
    return self.load_tensorflow_image('MAP2_CONFOCAL', 'map2_confocal.png')

  def load_nfh(self) -> lt.LabeledTensor:
    return self.load_tensorflow_image('NFH_CONFOCAL', 'nfh_confocal.png')

  def load_small_dic(self) -> lt.LabeledTensor:
    image_lt = self.load_dic()[0, :, :, 0]
    return image_lt[:256, :256]

  def load_small_dapi(self) -> lt.LabeledTensor:
    image_lt = self.load_dapi()[0, :, :, 0]
    return image_lt[:256, :256]

  def recordio_path(self) -> str:
    return self.data_path('TrainTensorflowExamples-00962-of-01024')

  def get_images(self, prefix: str, labeled_tensor: lt.LabeledTensor
                ) -> Tuple[List[str], lt.LabeledTensor]:
    labeled_axes = []
    for a in labeled_tensor.axes.values():
      if a.labels is not None:
        labeled_axes.append(a)
      else:
        labeled_axes.append((a.name, range(len(a))))

    labeled_tensor = lt.LabeledTensor(labeled_tensor.tensor, labeled_axes)

    iter_axis_names = []
    iter_axis_labels = []
    for a in labeled_tensor.axes.values():
      if a.name != 'row' and a.name != 'column' and a.name != 'color':
        iter_axis_names.append(a.name)
        iter_axis_labels.append(list(a.labels))

    filenames = []
    image_lts = []
    for labels in itertools.product(*iter_axis_labels):
      filename = [
          '%s_%s' % (name, l) for name, l in zip(iter_axis_names, labels)
      ]
      filename = prefix + '_'.join(filename) + '.png'
      filenames.append(filename)

      selection = {name: l for name, l in zip(iter_axis_names, labels)}
      select_lt = lt.select(labeled_tensor, selection)

      if len(select_lt.axes) == 2:
        select_lt = lt.transpose(select_lt, ['row', 'column'])
      else:
        select_lt = lt.transpose(select_lt, ['row', 'column', 'color'])

      image_lts.append(select_lt)

    return filenames, image_lts

  def profile(self,
              tensors: List[Union[tf.Tensor, tf.Operation, lt.LabeledTensor]]):
    tensors = [
        t.tensor if isinstance(t, lt.LabeledTensor) else t for t in tensors
    ]

    run_metadata = tf.RunMetadata()
    sv = tf.train.Supervisor(graph=tensors[0].graph)
    sess = sv.PrepareSession()
    sv.StartQueueRunners(sess)

    results = sess.run(
        tensors,
        options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
        run_metadata=run_metadata)

    options = tf.contrib.tfprof.model_analyzer.PRINT_ALL_TIMING_MEMORY
    options['viz'] = True
    tf.contrib.tfprof.model_analyzer.print_model_analysis(
        tf.get_default_graph(), run_meta=run_metadata, tfprof_options=options)

    sv.Stop()

    return results

  def eval(self,
           tensors: List[Union[tf.Tensor, tf.Operation, lt.LabeledTensor]]):

    tensors = [
        t.tensor if isinstance(t, lt.LabeledTensor) else t for t in tensors
    ]

    sv = tf.train.Supervisor(graph=tensors[0].graph)
    sess = sv.PrepareSession()
    # sess.run([tf.initialize_all_variables()])
    sv.StartQueueRunners(sess)

    results = sess.run(tensors)

    sv.Stop()

    return results

  def save_images(self, directory: str,
                  pairs: List[Tuple[List[str], List[lt.LabeledTensor]]]):
    filenames = list(itertools.chain(*[p[0] for p in pairs]))
    image_lts = list(itertools.chain(*[p[1] for p in pairs]))

    images = self.eval(image_lts)

    path = os.path.join(os.environ['TEST_TMPDIR'], directory)
    if not gfile.Exists(path):
      gfile.MkDir(path)

    for f, i in zip(filenames, images):
      util.write_image(os.path.join(path, f), i)

  def assert_tensors_equal(self, tensor_0, tensor_1):
    [tensor_0_eval, tensor_1_eval] = self.eval([tensor_0, tensor_1])
    np.testing.assert_allclose(
        tensor_0_eval, tensor_1_eval, rtol=0.0001, atol=0.0001)

  def assert_labeled_tensors_equal(self, tensor_0, tensor_1):
    self.assertEqual(tensor_0.axes, tensor_1.axes)
    self.assert_tensors_equal(tensor_0.tensor, tensor_1.tensor)

  def assert_images_near(self, directory: str, only_check_size: bool = False):
    """Assert images in the golden directory match those in the test."""
    # We assume all images are pngs.
    glob = os.path.join(os.environ['TEST_SRCDIR'], 'isl/testdata', directory,
                        '*.png')
    golden_image_paths = gfile.Glob(glob)
    assert golden_image_paths, glob

    logging.info('Golden images for test match are: %s', golden_image_paths)

    for gip in golden_image_paths:
      test_image_path = os.path.join(os.environ['TEST_TMPDIR'], directory,
                                     os.path.basename(gip))
      assert gfile.Exists(
          test_image_path), "Test image doesn't exist: %s" % test_image_path

      golden = util.read_image(gip)
      test = util.read_image(test_image_path)

      if only_check_size:
        assert golden.shape == test.shape, (golden.shape, test.shape)
      else:
        np.testing.assert_allclose(golden, test, rtol=0.0001, atol=0.0001)
