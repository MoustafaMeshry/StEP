from PIL import Image
from absl import app
import cv2
import glob
import numpy as np
import os.path as osp
import pickle
import tensorflow as tf

from options import FLAGS as opts
import data
import losses
import networks
import style_loss
import utils


def avoid_same_scene(sorted_neighbors, filenames, k_max_nearest,
                     k_max_farthest):
  print('Avoiding same scene for timelapse dataset')
  scene_id = [int(f[-5:-4]) for f in filenames]
  for i in range(10):
      print(filenames[i], scene_id[i])
  print(len(sorted_neighbors))
  num_swaps = 0
  for i in range(len(sorted_neighbors)):
    j = 0
    swap_idx = 0
    while j < len(sorted_neighbors) and swap_idx < k_max_nearest:
      if scene_id[sorted_neighbors[i][j]] != scene_id[i]:
        temp = sorted_neighbors[i][swap_idx]
        sorted_neighbors[i][swap_idx] = sorted_neighbors[i][j]
        sorted_neighbors[i][j] = temp
        swap_idx += 1
        num_swaps += 1
      j += 1
  print('total swaps = %d' % num_swaps)
  return sorted_neighbors


def get_triplet_input_fn(dataset_path, dist_file_path=None, max_n_imgs=-1,
                         k_max_nearest=5, k_max_farthest=13):
  filenames = get_app_pretraining_image_set(dataset_path, max_n_imgs)
  print('DBG: obtained %d input filenames for triplet inputs' % len(filenames))
  print('DBG: Computing pairwise style distances:')
  if dist_file_path is not None and osp.exists(dist_file_path):
    print('*** Loading distance matrix from %s' % dist_file_path)
    with open(dist_file_path, 'rb') as f:
      dist_matrix = pickle.load(f)['dist_matrix']
      print('loaded a dist_matrix of shape: %s' % str(dist_matrix.shape))
  else:
    dist_matrix = style_loss.compute_pairwise_style_loss_v2(filenames)
    dist_dict = {'dist_matrix': dist_matrix}
    print('Saving distance matrix to %s' % dist_file_path)
    with open(dist_file_path, 'wb') as f:
      pickle.dump(dist_dict, f)

  # Sort neighbors for each anchor image
  num_imgs = len(dist_matrix)
  sorted_neighbors = [np.argsort(dist_matrix[ii, :]) for ii in range(num_imgs)]
  if opts.dataset_name == 'space_needle_timelapse':
    k_max_nearest *= 3
    k_max_farthest *= 6  # since outliers contribute with 8 crops
    sorted_neighbors = avoid_same_scene(sorted_neighbors, filenames,
                                        k_max_nearest, k_max_farthest)

  def triplet_input_fn(anchor_idx):
    # start from 1 to avoid getting the same image as its neighbor
    positive_neighbor_idx = np.random.randint(1, k_max_nearest + 1)
    negative_neighbor_idx = num_imgs - 1 - np.random.randint(0, k_max_farthest)
    positive_img_idx = sorted_neighbors[anchor_idx][positive_neighbor_idx]
    negative_img_idx = sorted_neighbors[anchor_idx][negative_neighbor_idx]
    # Read anchor image
    anchor_rgb_path = osp.join(dataset_path, filenames[anchor_idx])
    anchor_input = data.load_normalized_gt_image(anchor_rgb_path)
    anchor_input = anchor_input.astype(np.float32)
    # Read positive image
    positive_rgb_path = osp.join(dataset_path, filenames[positive_img_idx])
    positive_input = data.load_normalized_gt_image(positive_rgb_path)
    positive_input = positive_input.astype(np.float32)
    # Read negative image
    negative_rgb_path = osp.join(dataset_path, filenames[negative_img_idx])
    negative_input = data.load_normalized_gt_image(negative_rgb_path)
    negative_input = negative_input.astype(np.float32)
    # Return triplet
    return anchor_input, positive_input, negative_input

  return triplet_input_fn


def get_tf_triplet_dataset_iter(
    dataset_path, trainset_size, dist_file_path, batch_size=4,
    deterministic_flag=False, shuffle_buf_size=128, repeat_flag=True):
  """
  TODO
  """
  # Create a dataset of anchor image indices.
  idx_dataset = tf.data.Dataset.range(trainset_size)
  # Create a mapper function from anchor idx to triplet images.
  triplet_mapper = lambda idx: tuple(tf.py_func(
      get_triplet_input_fn(dataset_path, dist_file_path, trainset_size), [idx],
      [tf.float32, tf.float32, tf.float32]))
  # Convert triplet to a dictionary for the estimator input format.
  triplet_to_dict_mapper = lambda anchor, pos, neg: {
      'anchor_img': anchor, 'positive_img': pos, 'negative_img': neg}
  if repeat_flag:
    idx_dataset = idx_dataset.repeat()  # Repeat indefinitely.
  if not deterministic_flag:
    idx_dataset = idx_dataset.shuffle(shuffle_buf_size)
    triplet_dataset = idx_dataset.map(
        triplet_mapper, num_parallel_calls=max(4, batch_size // 4))
    triplet_dataset = triplet_dataset.map(
        triplet_to_dict_mapper, num_parallel_calls=max(4, batch_size // 4))
  else:
    triplet_dataset = idx_dataset.map(triplet_mapper, num_parallel_calls=None)
    triplet_dataset = triplet_dataset.map(triplet_to_dict_mapper,
                                          num_parallel_calls=None)
  triplet_dataset = triplet_dataset.batch(batch_size)
  if not deterministic_flag:
    triplet_dataset = triplet_dataset.prefetch(4)  # Prefetch a few batches.
  return triplet_dataset.make_one_shot_iterator()


def get_random_z(batch_size, mean=0, std=1):
  return tf.random_normal(shape=[batch_size, 1, 1, opts.app_vector_size],
                          mean=mean, stddev=std, dtype=tf.float32)


def build_model_fn(batch_size, lr_app_pretrain=0.0001, adam_beta1=0.0,
                   adam_beta2=0.99):
  def model_fn(features, labels, mode, params):
    del labels, params

    step = tf.train.get_global_step()
    assert opts.use_concat
    app_func = networks.DRITAppearanceEncoderConcat(
      'appearance_net', opts.appearance_nc, opts.normalize_drit_Ez)

    if mode == tf.estimator.ModeKeys.TRAIN:
      op_increment_step = tf.assign_add(step, 1)
      with tf.name_scope('Appearance_Loss'):
        anchor_img = features['anchor_img']
        positive_img = features['positive_img']
        negative_img = features['negative_img']
        # Compute embeddings (each of shape [batch_sz, 1, 1, app_vector_sz])
        # Squeeze into shape of [batch_sz x vec_sz]
        z_anchor, mu_anchor, logvar_anchor = app_func(anchor_img)
        z_pos, mu_pos, logvar_pos = app_func(positive_img)
        z_neg, mu_neg, logvar_neg = app_func(negative_img)
        anchor_embedding = tf.squeeze(z_anchor, axis=[1, 2], name='z_anchor')
        positive_embedding = tf.squeeze(z_pos, axis=[1, 2])
        negative_embedding = tf.squeeze(z_neg, axis=[1, 2])
        # Compute triplet loss
        margin = 0.1
        anchor_positive_dist = tf.reduce_sum(
            tf.square(anchor_embedding - positive_embedding), axis=1)
        anchor_negative_dist = tf.reduce_sum(
            tf.square(anchor_embedding - negative_embedding), axis=1)
        triplet_loss = anchor_positive_dist - anchor_negative_dist + margin
        triplet_loss = tf.maximum(triplet_loss, 0.)
        triplet_loss = tf.reduce_sum(triplet_loss) / batch_size
        if opts.use_concat:
          if opts.w_loss_z_l2 > 0:
            l2_reg = losses.l2_regularize(z_anchor) + \
                losses.l2_regularize(z_pos) + losses.l2_regularize(z_neg)
            regularizer = opts.w_loss_z_l2 * l2_reg
            reg_name = 'l2_reg'
        else:
          l2_reg = losses.l2_regularize(z_anchor) + \
              losses.l2_regularize(z_pos) + losses.l2_regularize(z_neg)
          regularizer = opts.w_loss_z_l2 * l2_reg
          reg_name = 'l2_reg'
        total_loss = triplet_loss + regularizer
        tf.summary.scalar('appearance_triplet_loss', triplet_loss)
        tf.summary.scalar(reg_name, regularizer)

        # Image summaries
        tb_vis = tf.concat([anchor_img, positive_img, negative_img], axis=2)
        with tf.name_scope('triplet_vis'):
          tf.summary.image('anchor-pos-neg', tb_vis)

      optimizer = tf.train.AdamOptimizer(lr_app_pretrain, adam_beta1,
                                         adam_beta2)
      optimizer = tf.contrib.estimator.TowerOptimizer(optimizer)
      app_vars = utils.model_vars('appearance_net')[0]
      print('Number of variables in the style encoder = %d' % len(app_vars))
      for ii, v in enumerate(app_vars):
        print('%03d) %s' % (ii, str(v)))
      app_train_op = optimizer.minimize(total_loss, var_list=app_vars)
      return tf.estimator.EstimatorSpec(
          mode=mode, loss=total_loss,
          train_op=tf.group(app_train_op, op_increment_step))
    elif mode == tf.estimator.ModeKeys.PREDICT:
      imgs = features['anchor_img']
      embeddings = tf.squeeze(app_func(imgs), axis=[1, 2])
      app_vars = utils.model_vars('appearance_net')[0]
      tf.train.init_from_checkpoint(osp.join(opts.train_dir),
                                    {'appearance_net/': 'appearance_net/'})
      return tf.estimator.EstimatorSpec(mode=mode, predictions=embeddings)
    else:
      raise ValueError('Unsupported mode for the appearance model: ' + mode)

  return model_fn


def get_app_pretraining_image_set(imageset_dir, max_n_imgs):
  images_paths = sorted(glob.glob(osp.join(imageset_dir, '*.jpg')))
  if max_n_imgs > -1 and max_n_imgs < len(images_paths):
    np.random.seed(0)
    indices = sorted(np.random.permutation(len(images_paths))[:max_n_imgs])
    images_paths = [images_paths[ii] for ii in indices]
  return images_paths


def compute_dist_matrix(imageset_dir, dist_file_path, max_n_imgs=-1,
                        recompute_dist=False):
  if not recompute_dist and osp.exists(dist_file_path):
   print('*** Loading distance matrix from %s' % dist_file_path)
   with open(dist_file_path, 'rb') as f:
     dist_matrix = pickle.load(f)['dist_matrix']
     print('loaded a dist_matrix of shape: %s' % str(dist_matrix.shape))
     return dist_matrix
  else:
    images_paths = get_app_pretraining_image_set(imageset_dir, max_n_imgs)
    dist_matrix = style_loss.compute_pairwise_style_loss_v2(images_paths)
    dist_dict = {'dist_matrix': dist_matrix}
    print('Saving distance matrix to %s' % dist_file_path)
    with open(dist_file_path, 'wb') as f:
      pickle.dump(dist_dict, f)
    return dist_matrix


def train_appearance(train_dir, imageset_dir, dist_file_path, max_n_imgs=-1):
  """
  TODO
  """
  batch_size = 12
  lr_app_pretrain = 0.001

  trainset_size = len(glob.glob(osp.join(imageset_dir, '*.jpg')))
  if max_n_imgs > -1 and max_n_imgs < trainset_size:
    trainset_size = max_n_imgs
  resume_step = utils.load_global_step_from_checkpoint_dir(train_dir)
  if resume_step != 0:
    tf.logging.warning('DBG: resuming apperance pretraining at %d!' %
                       resume_step)
  model_fn = build_model_fn(batch_size, lr_app_pretrain)
  config = tf.estimator.RunConfig(
      save_summary_steps=50,
      save_checkpoints_steps=500,
      keep_checkpoint_max=5,
      log_step_count_steps=100)
  est = tf.estimator.Estimator(
      tf.contrib.estimator.replicate_model_fn(model_fn), train_dir,
      config, params={})
  # Get input function
  input_train_fn = lambda: get_tf_triplet_dataset_iter(
      imageset_dir, trainset_size, dist_file_path,
      batch_size=batch_size).get_next()
  print('Starting pretraining steps...')
  # Train indefinitely, kill the training when the loss curve saturates.
  est.train(input_train_fn, steps=None, hooks=None)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  train_dir = opts.pretrain_dir
  dataset_name = opts.dataset_name
  imageset_dir = opts.imageset_dir
  max_n_imgs = opts.max_app_images
  output_dir = opts.metadata_output_dir
  if not osp.exists(output_dir):
    os.makedirs(output_dir)
  dist_file_path = osp.join(output_dir, 'dist_%s.pckl' % dataset_name)
  compute_dist_matrix(imageset_dir, dist_file_path, max_n_imgs)
  train_appearance(train_dir, imageset_dir, dist_file_path, max_n_imgs)

if __name__ == '__main__':
  app.run(main)
