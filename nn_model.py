# Copyright 2018 The TensorFlow Probability Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import warnings

# Dependency imports
from absl import flags
# import matplotlib
# matplotlib.use("Agg")
# from matplotlib import figure  
# import matplotlib.pylab as plt
# from matplotlib.backends import backend_agg
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from glob import glob

try:
  import seaborn as sns  
  HAS_SEABORN = True
except ImportError:
  HAS_SEABORN = False

tfd = tfp.distributions
flags.DEFINE_integer("output_classes",
                      default=62,
                      help="Number of output panels")
flags.DEFINE_float("learning_rate",
                   default=0.0025,
                   help="Initial learning rate.")
flags.DEFINE_integer("max_steps",
                     default=5000,
                     help="Number of training steps to run.")
flags.DEFINE_integer("batch_size",
                     default=2048,
                     help="Batch size.")
flags.DEFINE_string("data_file",
                    default="./panel_data.npz",
                    help="Prepared file where data is stored (makeDatasets.py).")
flags.DEFINE_string("model_dir",default="../model",
                    help="Directory to put the model's fit.")
flags.DEFINE_integer("viz_steps",
                     default=400,
                     help="Frequency at which save visualizations.")
flags.DEFINE_integer("num_monte_carlo",
                     default=500,
                     help="Network draws to compute predictive probabilities.")

FLAGS = flags.FLAGS


def plot_weight_posteriors(names, qm_vals, qs_vals, fname):
  """Save a PNG plot with histograms of weight means and stddevs.

  Args:
    names: A Python `iterable` of `str` variable names.
    qm_vals: A Python `iterable`, the same length as `names`,
      whose elements are Numpy `array`s, of any shape, containing
      posterior means of weight varibles.
    qs_vals: A Python `iterable`, the same length as `names`,
      whose elements are Numpy `array`s, of any shape, containing
      posterior standard deviations of weight varibles.
    fname: Python `str` filename to save the plot to.
  """
  fig = figure.Figure(figsize=(6, 3))
  canvas = backend_agg.FigureCanvasAgg(fig)

  ax = fig.add_subplot(1, 2, 1)
  for n, qm in zip(names, qm_vals):
    sns.distplot(qm.flatten(), ax=ax, label=n)
  ax.set_title("weight means")
  ax.set_xlim([-1.5, 1.5])
  ax.legend()

  ax = fig.add_subplot(1, 2, 2)
  for n, qs in zip(names, qs_vals):
    sns.distplot(qs.flatten(), ax=ax)
  ax.set_title("weight stddevs")
  ax.set_xlim([0, 1.])

  fig.tight_layout()
  canvas.print_figure(fname, format="png")
  print("saved {}".format(fname))


def plot_heldout_prediction(input_vals, probs,
                            fname, n=10, title=""):
  """Save a PNG plot visualizing posterior uncertainty on heldout data.

  Args:
    input_vals: A `float`-like Numpy `array` of shape
      `[num_heldout] + IMAGE_SHAPE`, containing heldout input images.
    probs: A `float`-like Numpy array of shape `[num_monte_carlo,
      num_heldout, num_classes]` containing Monte Carlo samples of
      class probabilities for each heldout sample.
    fname: Python `str` filename to save the plot to.
    n: Python `int` number of datapoints to vizualize.
    title: Python `str` title for the plot.
  """
  fig = figure.Figure(figsize=(9, 3*n))
  canvas = backend_agg.FigureCanvasAgg(fig)
  for i in range(n):
    ax = fig.add_subplot(n, 3, 3*i + 1)
    # ax.imshow(input_vals[i, :].reshape(IMAGE_SHAPE[:-1]), interpolation="None")

    ax = fig.add_subplot(n, 3, 3*i + 2)
    for prob_sample in probs:
      sns.barplot(np.arange(10), prob_sample[i, :], alpha=0.1, ax=ax)
      ax.set_ylim([0, 1])
    ax.set_title("posterior samples")

    ax = fig.add_subplot(n, 3, 3*i + 3)
    sns.barplot(np.arange(10), np.mean(probs[:, i, :], axis=0), ax=ax)
    ax.set_ylim([0, 1])
    ax.set_title("predictive probs")
  fig.suptitle(title)
  fig.tight_layout()

  canvas.print_figure(fname, format="png")
  print("saved {}".format(fname))

def get_model():
  return(tf.keras.Sequential([
    # tfp.layers.DenseFlipout(1024),#, activation=tf.nn.relu),
    tfp.layers.DenseFlipout(FLAGS.output_classes) 
    ]))


def build_input_pipeline(panel_data, batch_size, heldout_size):
  """Build an Iterator switching between train and heldout data."""

  # Build an iterator over training batches.
  training_dataset = tf.data.Dataset.from_tensor_slices(
      (np.float32(panel_data['xTrain']), np.int32(panel_data['yTrain'])))
  training_batches = training_dataset.shuffle(
      50000, reshuffle_each_iteration=True).repeat().batch(batch_size)
  training_iterator = training_batches.make_one_shot_iterator()

  # Build a iterator over the heldout set with batch_size=heldout_size,
  # i.e., return the entire heldout set as a constant.
  heldout_dataset = tf.data.Dataset.from_tensor_slices(
      (np.float32(panel_data['xVal']),np.int32(panel_data['yVal'])))
  heldout_frozen = (heldout_dataset.take(heldout_size).
                    repeat().batch(heldout_size))
  heldout_iterator = heldout_frozen.make_one_shot_iterator()

  # Build a iterator over the test set with batch_size=test_size,
  # i.e., return the entire test set as a constant.
  test_size=panel_data['yTest'].shape[0]
  test_dataset = tf.data.Dataset.from_tensor_slices(
      (np.float32(panel_data['xTest']),np.int32(panel_data['yTest'])))
  test_frozen = (test_dataset.take(test_size).repeat().batch(test_size))
  test_iterator = test_frozen.make_one_shot_iterator()


  # Combine these into a feedable iterator that can switch between training
  # and validation inputs.
  handle = tf.placeholder(tf.string, shape=[])
  feedable_iterator = tf.data.Iterator.from_string_handle(
      handle, training_batches.output_types, training_batches.output_shapes)
  images, labels = feedable_iterator.get_next()

  return images, labels, handle, training_iterator, heldout_iterator, test_iterator

def read_panels_fv(datafile):
  return(np.load(datafile))


