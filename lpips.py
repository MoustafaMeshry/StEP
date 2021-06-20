import os
import os.path as osp
import sys

import tensorflow as tf
from six.moves import urllib

class LPIPS:
  def __init__(self, input0, input1, model='net-lin', net='alex', version=0.1):
    """
    Learned Perceptual Image Patch Similarity (LPIPS) metric.

    Args:
        input0: An image tensor of shape [NHWC], with values in [-1, 1].
        input1: An image tensor of shape [NHWC], with values in [-1, 1].

    Returns:
        The Learned Perceptual Image Patch Similarity (LPIPS) distance.

    Reference:
        Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, Oliver Wang.
        The Unreasonable Effectiveness of Deep Features as a Perceptual Metric.
        In CVPR, 2018.
    """
    # flatten the leading dimensions
    batch_shape = tf.shape(input0)[:-3]
    input0 = tf.reshape(input0, tf.concat([[-1], tf.shape(input0)[-3:]], axis=0))
    input1 = tf.reshape(input1, tf.concat([[-1], tf.shape(input1)[-3:]], axis=0))
    input0 = tf.transpose(input0, [0, 3, 1, 2])
    input1 = tf.transpose(input1, [0, 3, 1, 2])

    input0_name, input1_name = '0:0', '1:0'

    default_graph = tf.get_default_graph()
    producer_version = default_graph.graph_def_versions.producer
    producer_version = 27  # 38

    cache_dir = 'lpips_pb'
    pb_fname = '%s_%s_v%s_%d.pb' % (model, net, version, producer_version)

    with open(os.path.join(cache_dir, pb_fname), 'rb') as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())
      _ = tf.import_graph_def(graph_def,
                              input_map={input0_name: input0, input1_name: input1})
      distance, = default_graph.get_operations()[-1].outputs
      assert distance.name[-12:] == 'Reshape_10:0', distance.name

    if distance.shape.ndims == 4:
      distance = tf.squeeze(distance, axis=[-3, -2, -1])
    distance = tf.reshape(distance, batch_shape)
    self.distance = tf.reduce_mean(distance)

  def __call__(self):
    return self.distance
