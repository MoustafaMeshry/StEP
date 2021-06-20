"""Utilities for GANs.

Basic functions such as generating sample grid, exporting to PNG, etc...
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os.path
import time

from PIL import Image
import numpy as np
import tensorflow as tf


def crop_to_multiple(img, size_multiple):
  """ Crops the image so that its dimensions are multiples of size_multiple."""
  new_width = (img.shape[1] // size_multiple) * size_multiple
  new_height = (img.shape[0] // size_multiple) * size_multiple
  offset_x = (img.shape[1] - new_width) // 2
  offset_y = (img.shape[0] - new_height) // 2
  return img[offset_y:offset_y + new_height, offset_x:offset_x + new_width, :]


def load_global_step_from_checkpoint_dir(checkpoint_dir):
  """Loads  the global step from the checkpoint directory.

  Args:
    checkpoint_dir: string, path to the checkpoint directory.

  Returns:
    int, the global step of the latest checkpoint or 0 if none was found.
  """
  try:
    checkpoint_reader = tf.train.NewCheckpointReader(
        tf.train.latest_checkpoint(checkpoint_dir))
    return checkpoint_reader.get_tensor(tf.GraphKeys.GLOBAL_STEP)
  except:  # pylint: disable=bare-except
    return 0


def model_vars(prefix):
  """Return trainable variables matching a prefix.

  Args:
    prefix: string, the prefix variable names must match.

  Returns:
    a tuple (match, others) of TF variables, 'match' contains the matched
     variables and 'others' contains the remaining variables.
  """
  match, no_match = [], []
  for x in tf.trainable_variables():
    if x.name.startswith(prefix):
      match.append(x)
    else:
      no_match.append(x)
  return match, no_match


def average_gradients(tower_grads):
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(grads, 0)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads


# By default, all variables will be placed on '/gpu:0'
# So we need a custom device function, to assign all variables to '/cpu:0'
# Note: If GPUs are peered, '/gpu:0' can be a faster option
PS_OPS = ['Variable', 'VariableV2', 'AutoReloadVariable']

def assign_to_device(device, ps_device='/cpu:0'):
  def _assign(op):
    node_def = op if isinstance(op, tf.NodeDef) else op.node_def
    if node_def.op in PS_OPS:
      return "/" + ps_device
    else:
      return device

  return _assign


def to_png(x):
  """Convert a 3D tensor to png.

  Args:
    x: Tensor, 01C formatted input image.

  Returns:
    Tensor, 1D string representing the image in png format.
  """
  with tf.Graph().as_default():
    with tf.Session() as sess_temp:
      x = tf.constant(x)
      y = tf.image.encode_png(
          tf.cast(
              tf.clip_by_value(tf.round(127.5 + 127.5 * x), 0, 255), tf.uint8),
          compression=9)
      return sess_temp.run(y)


def images_to_grid(images):
  """Converts a grid of images (5D tensor) to a single image.

  Args:
    images: 5D tensor (count_y, count_x, height, width, colors), grid of images.

  Returns:
    a 3D tensor image of shape (count_y * height, count_x * width, colors).
  """
  ny, nx, h, w, c = images.shape
  images = images.transpose(0, 2, 1, 3, 4)
  images = images.reshape([ny * h, nx * w, c])
  return images


def save_images(image, output_dir, cur_nimg):
  """Saves images to disk.

  Saves a file called 'name.png' containing the latest samples from the
   generator and a file called 'name_123.png' where 123 is the KiB of trained
   images.

  Args:
    image: 3D numpy array (height, width, colors), the image to save.
    output_dir: string, the directory where to save the image.
    cur_nimg: int, current number of images seen by training.

  Returns:
    None
  """
  for name in ('name.png', 'name_%06d.png' % (cur_nimg >> 10)):
    with tf.gfile.Open(os.path.join(output_dir, name), 'wb') as f:
      f.write(image)


class HookReport(tf.train.SessionRunHook):
  """Custom reporting hook.

  Register your tensor scalars with HookReport.log_tensor(my_tensor, 'my_name').
  This hook will report their average values over report period argument
  provided to the constructed. The values are printed in the order the tensors
  were registered.

  Attributes:
    step: int, the current global step.
    active: bool, whether logging is active or disabled.
  """
  _REPORT_KEY = 'report'
  _TENSOR_NAMES = {}

  def __init__(self, period, batch_size):
    self.step = 0
    self.active = True
    self._period = period // batch_size
    self._batch_size = batch_size
    self._sums = np.array([])
    self._count = 0
    self._nimgs_per_cycle = 0
    self._step_ratio = 0
    self._start = time.time()
    self._nimgs = 0
    self._batch_size = batch_size

  def disable(self):
    parent = self

    class Disabler(object):

      def __enter__(self):
        parent.active = False
        return parent

      def __exit__(self, exc_type, exc_val, exc_tb):
        parent.active = True

    return Disabler()

  def begin(self):
    self.active = True
    self._count = 0
    self._nimgs_per_cycle = 0
    self._start = time.time()

  def before_run(self, run_context):
    if not self.active:
      return
    del run_context
    fetches = tf.get_collection(self._REPORT_KEY)
    return tf.train.SessionRunArgs(fetches)

  def after_run(self, run_context, run_values):
    if not self.active:
      return
    del run_context
    results = run_values.results
    # Note: sometimes the returned step is incorrect (off by one) for some
    # unknown reason.
    self.step = results[-1] + 1
    self._count += 1
    self._nimgs_per_cycle += self._batch_size
    self._nimgs += self._batch_size

    if not self._sums.size:
      self._sums = np.array(results[:-1], 'd')
    else:
      self._sums += np.array(results[:-1], 'd')

    if self.step // self._period != self._step_ratio:
      fetches = tf.get_collection(self._REPORT_KEY)[:-1]
      stats = '  '.join('%s=% .2f' % (self._TENSOR_NAMES[tensor],
                                      value / self._count)
                        for tensor, value in zip(fetches, self._sums))
      stop = time.time()
      tf.logging.info('step=%d, kimg=%d  %s  [%.2f img/s]' %
                      (self.step, ((self.step * self._batch_size) >> 10),
                       stats, self._nimgs_per_cycle / (stop - self._start)))
      self._step_ratio = self.step // self._period
      self._start = stop
      self._sums *= 0
      self._count = 0
      self._nimgs_per_cycle = 0

  def end(self, session=None):
    del session

  @classmethod
  def log_tensor(cls, tensor, name):
    """Adds a tensor to be reported by the hook.

    Args:
      tensor: `tensor scalar`, a value to report.
      name: string, the name to give the value in the report.

    Returns:
      None.
    """
    cls._TENSOR_NAMES[tensor] = name
    tf.add_to_collection(cls._REPORT_KEY, tensor)
                                                                                

def save_normalized_image(normalized_img, save_path):
  img = np.squeeze(normalized_img)
  img = (img + 1) * 255./2  # convert range [-1, 1] into [0, 255]
  img = img.astype('uint8')
  Image.fromarray(img).save(save_path)
