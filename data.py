from options import FLAGS as opts
from PIL import Image
import glob
import numpy as np
import os.path as osp
import random
import tensorflow as tf


def provide_data(dataset_name='', parent_dir='', batch_size=8, subset=None,
                 repeat_flag=False, max_examples=None, shuffle=128,
                 data_aug=False, crop_size=256):
  input_iter = provide_data_iter(
      dataset_name=dataset_name, parent_dir=parent_dir, batch_size=batch_size,
      subset=subset, repeat_flag=repeat_flag, max_examples=max_examples,
      shuffle=shuffle, data_aug=data_aug, crop_size=crop_size)
  input_dict_var = input_iter.get_next()
  return input_dict_var


def provide_data_iter(dataset_name='', parent_dir='', batch_size=8, subset=None,
                      repeat_flag=False, max_examples=None, shuffle=128,
                      data_aug=False, crop_size=256):
  """
  TODO:
    seeds: list, {None means random crops, a list with a single seed will be
    treated as a single central crop (seed value will be ignore), otherwise, the
    provided N seeds will be used to generate N fixed crops.}
  """
  dataset_dir = osp.join(parent_dir, dataset_name)
  if subset is not None:
    dataset_dir = osp.join(dataset_dir, subset)
  # Create a mapper function from anchor idx to triplet images.
  parse_mapper = lambda idx: tuple(tf.py_func(
      get_input_fn(dataset_dir, data_aug, crop_size), [idx],
                   [tf.float32, tf.float32]))
  # Convert triplet to a dictionary for the estimator input format.
  tuple_to_dict_mapper = lambda x_in, x_gt: {
      'conditional_input': x_in, 'expected_output': x_gt}

  input_dict_iter = multi_input_fn_record(
      dataset_dir, parse_mapper, tuple_to_dict_mapper, batch_size, repeat_flag,
      max_examples, shuffle)
  return input_dict_iter


def _list_image_dir(image_dir_path, sort=True,
                    extensions=('jpg', 'JPG', 'png', 'PNG')):
  filenames = []
  for ext in extensions:
    filenames.extend(glob.glob(osp.join(image_dir_path, '*.%s' % ext)))
  if sort:
    filenames = sorted(filenames)
  return filenames


def multi_input_fn_record(
    dataset_dir, parse_mapper, tuple_to_dict_mapper, batch_size,
    repeat_flag=False, max_examples=None, shuffle=128):
  """Creates a Dataset pipeline for tfrecord files.

  Args:
    batch_size: int, batch size.
    shuffle: int, the size of the shuffle buffer. If 0, then no shufling at all.

  Returns:
    Dataset iterator.
  """
  filenames = _list_image_dir(dataset_dir)
  deterministic_flag = shuffle == 0
  trainset_size = len(filenames)
  assert trainset_size > 0, ('Error! input pattern "%s" didn\'t match any '
                              'files' % input_pattern)
  # Create a dataset of image indices.
  idx_dataset = tf.data.Dataset.range(trainset_size)

  if repeat_flag:
    idx_dataset = idx_dataset.repeat()  # Repeat indefinitely.
  if not deterministic_flag:
    idx_dataset = idx_dataset.shuffle(shuffle)
    dataset = idx_dataset.map(
        parse_mapper, num_parallel_calls=max(4, batch_size // 4))
    dataset = dataset.map(
        tuple_to_dict_mapper, num_parallel_calls=max(4, batch_size // 4))
  else:
    dataset = idx_dataset.map(parse_mapper, num_parallel_calls=None)
    dataset = dataset.map(tuple_to_dict_mapper, num_parallel_calls=None)
  # Prepare paired inputs for the DRIT pipeline
  use_cross_cycle = opts.loss_cyclic_recon or opts.loss_D_cyclic or opts.loss_D_swap_z or opts.loss_z_recon
  if (use_cross_cycle or opts.training_pipeline == 'drit') and not deterministic_flag:
    dataset1 = dataset.shuffle(shuffle)
    dataset2 = dataset.shuffle(shuffle)
    paired_dataset = tf.data.Dataset.zip((dataset1, dataset2))

    def _join_paired_dataset(features_a, features_b):
      features_a['conditional_input_2'] = features_b['conditional_input']
      features_a['expected_output_2'] = features_b['expected_output']
      return features_a

    joined_dataset = paired_dataset.map(_join_paired_dataset)
    dataset = joined_dataset

  if max_examples is not None:
    dataset = dataset.take(max_examples)
  dataset = dataset.batch(batch_size)
  if not deterministic_flag:
    dataset = dataset.prefetch(24)  # Prefetch a few batches.

  return dataset.make_one_shot_iterator()


def load_normalized_image(img_path, rescale_coin=0):
  img = Image.open(img_path)
  if rescale_coin > 0.5:
    img = img.resize((320*2, 320))
  elif rescale_coin > 0.25 and opts.dataset_name == 'facades':
    img = img.resize((384*2, 384))
  rgb = np.array(img)
  rgb_normalized = rgb * 2. / 255. - 1
  return rgb_normalized 


def load_normalized_input_image(img_path, rescale_coin=0):
  rgb_normalized = load_normalized_image(img_path, rescale_coin=rescale_coin)
  width = rgb_normalized.shape[1]
  if opts.dataset_name in ['edges2shoes', 'edges2handbags', 'night2day', 'KDEF_pair1_filtered']:
    inp_img = rgb_normalized[:, :width//2, :]
  elif opts.dataset_name in ['maps', 'facades', 'space_needle_timelapse']:
    inp_img = rgb_normalized[:, width//2:, :]
  else:
    raise ValueError('The "%s" dataset is not supported' % opts.dataset_name)
  return inp_img


def load_normalized_gt_image(img_path, rescale_coin=0):
  rgb_normalized = load_normalized_image(img_path, rescale_coin=rescale_coin)
  width = rgb_normalized.shape[1]
  if opts.dataset_name in ['edges2shoes', 'edges2handbags', 'night2day', 'KDEF_pair1_filtered']:
    gt_img = rgb_normalized[:, width//2:, :]
  elif opts.dataset_name in ['maps', 'facades', 'space_needle_timelapse']:
    gt_img = rgb_normalized[:, :width//2, :]
  elif opts.dataset_name in ['celeba128', 'KDEF_crop256']:
    gt_img = rgb_normalized
  else:
    raise ValueError('The "%s" dataset is not supported' % opts.dataset_name)
  return gt_img


def save_normalized_image(img, img_path):
  rgb = (img + 1) * 255. / 2.
  rgb = rgb.astype('uint8')
  Image.fromarray(rgb).save(img_path)


def get_input_fn(dataset_path, data_aug=False, crop_size=256):
  filenames = _list_image_dir(dataset_path)
  if data_aug and opts.run_mode == 'train':
      random.shuffle(filenames)
  print('DBG: obtained %d input filenames for inputs' % len(filenames))

  def input_fn(img_idx):
    # Read anchor image
    img_path = osp.join(dataset_path, filenames[img_idx])
    if data_aug and opts.dataset_name in ['facades', 'night2day']:
      rescale_coin = random.uniform(0, 1)
    else:
      rescale_coin = 0
    x_in = load_normalized_input_image(img_path, rescale_coin)
    x_gt = load_normalized_gt_image(img_path, rescale_coin)
    w, h, _ = x_gt.shape
    assert crop_size <= min(h, w), (
      'image size %dx%d is smaller than a crop_size of %d' % (h, w, crop_size))
    if w > crop_size or h > crop_size:
      if data_aug:
        st_y = random.randint(0, h - crop_size)
        st_x = random.randint(0, w - crop_size)
      else:  # center crop
        st_y = (h - crop_size) // 2
        st_x = (w - crop_size) // 2
      x_in = x_in[st_y:st_y+crop_size, st_x:st_x+crop_size, :]
      x_gt = x_gt[st_y:st_y+crop_size, st_x:st_x+crop_size, :]
    if data_aug and opts.flip_horizontal:
      coin = random.uniform(0, 1)
      if coin < 0.5:
        x_in = np.fliplr(x_in)
        x_gt = np.fliplr(x_gt)
    x_in = x_in.astype(np.float32)
    x_gt = x_gt.astype(np.float32)
    return x_in, x_gt

  return input_fn
