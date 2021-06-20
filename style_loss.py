from PIL import Image
import numpy as np
import tensorflow as tf

from options import FLAGS as opts
import data
import layers
import vgg16


def gram_matrix(layer):
  """Computes the gram_matrix for a batch of single vgg layer
  Input:
    layer: a batch of vgg activations for a single conv layer
  Returns:
    gram: [batch_sz x num_channels x num_channels]: a batch of gram matrices
  """
  batch_size, height, width, num_channels = layer.get_shape().as_list()
  features = tf.reshape(layer, [batch_size, height * width, num_channels])
  num_elements = tf.constant(num_channels * height * width, tf.float32)
  gram = tf.matmul(features, features, adjoint_a=True) / num_elements
  return gram


def compute_gram_matrices(
    images, vgg_layers=['conv1_2', 'conv2_2', 'conv3_2', 'conv4_2', 'conv5_2']):
  """Computes the gram matrix representation of a batch of images"""
  vgg_net = vgg16.Vgg16(opts.vgg16_path)
  vgg_acts = vgg_net.get_vgg_activations(images, vgg_layers)
  grams = [gram_matrix(layer) for layer in vgg_acts]
  return grams


def compute_pairwise_style_loss_v2(image_paths_list):
  grams_all = [None] * len(image_paths_list)
  crop_height, crop_width = opts.crop_size, opts.crop_size
  img_var = tf.placeholder(tf.float32, shape=[1, crop_height, crop_width, 3])
  vgg_layers = ['conv%d_2' % i for i in range(1, 6)]  # conv1 through conv5
  grams_ops = compute_gram_matrices(img_var, vgg_layers)
  with tf.Session() as sess:
    # sess.run(init_op)
    for ii, img_path in enumerate(image_paths_list):
      print('Computing gram matrices for image #%d' % (ii + 1))
      img = data.load_normalized_gt_image(img_path)
      img = img.astype(np.float32)
      img = np.expand_dims(img, axis=0)
      grams_all[ii] = sess.run(grams_ops, feed_dict={img_var: img})
  print('Number of images = %d' % len(grams_all))
  print('Gram matrices per image:')
  for i in range(len(grams_all[0])):
    print('gram_matrix[%d].shape = %s' % (i, grams_all[0][i].shape))
  n_imgs = len(grams_all)
  dist_matrix = np.zeros((n_imgs, n_imgs))
  for i in range(n_imgs):
    print('Computing distances for image #%d' % i)
    for j in range(i + 1, n_imgs):
      loss_style = 0
      # Compute loss using all gram matrices from all layers
      for gram_i, gram_j in zip(grams_all[i], grams_all[j]):
        loss_style += np.mean((gram_i - gram_j) ** 2, axis=(1, 2))
      # # Compute loss using gram matrices at the shallowest layer only
      # loss_style = np.mean((grams_all[i][0] - grams_all[j][0]) ** 2, axis=(1, 2))
      # Compute loss using gram matrices at the shallowest layer only
      # loss_style = np.mean((grams_all[i][-1] - grams_all[j][-1]) ** 2, axis=(1, 2))
      dist_matrix[i][j] = dist_matrix[j][i] = loss_style

  return dist_matrix
