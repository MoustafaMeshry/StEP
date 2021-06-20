from PIL import Image
from absl import app
from absl import flags
from options import FLAGS as opts
import datetime
import glob
import losses
import lpips
import numpy as np
import os
import os.path as osp
import skimage.measure
import tensorflow as tf

FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'edges2shoes', "Ddataset name")
flags.DEFINE_string('exp_title', None, "experiment title")


def crop_to_multiple(img, size_multiple=64):
  new_width = (img.shape[1] // size_multiple) * size_multiple
  new_height = (img.shape[0] // size_multiple) * size_multiple
  offset_x = (img.shape[1] - new_width) // 2
  offset_y = (img.shape[0] - new_height) // 2
  return img[offset_y:offset_y + new_height, offset_x:offset_x + new_width, :]


def compute_l1_loss_metric(image_set1_paths, image_set2_paths):
  assert len(image_set1_paths) == len(image_set2_paths)
  assert len(image_set1_paths) > 0
  print('Evaluating L1 loss for %d pairs' % len(image_set1_paths))

  total_loss = 0.
  for ii, (img1_path, img2_path) in enumerate(zip(image_set1_paths,
                                                  image_set2_paths)):
    conc_img_ar = np.array(Image.open(img1_path))
    img_width = conc_img_ar.shape[1]
    img1_in_ar = conc_img_ar[:, img_width//3:-img_width//3, :]
    img2_in_ar = conc_img_ar[:, -img_width//3:, :]

    img1_in_ar = np.expand_dims(img1_in_ar, axis=0)
    img2_in_ar = np.expand_dims(img2_in_ar, axis=0)

    loss_l1 = np.mean(np.abs(img1_in_ar - img2_in_ar))
    total_loss += loss_l1

  return total_loss / len(image_set1_paths)


def compute_psnr_loss_metric(image_set1_paths, image_set2_paths):
  assert len(image_set1_paths) == len(image_set2_paths)
  assert len(image_set1_paths) > 0
  print('Evaluating PSNR loss for %d pairs' % len(image_set1_paths))

  total_loss = 0.
  for ii, (img1_path, img2_path) in enumerate(zip(image_set1_paths,
                                                  image_set2_paths)):
    conc_img_ar = np.array(Image.open(img1_path))
    img_width = conc_img_ar.shape[1]
    img1_in_ar = conc_img_ar[:, img_width//3:-img_width//3, :]
    img2_in_ar = conc_img_ar[:, -img_width//3:, :]

    img1_in_ar = np.expand_dims(img1_in_ar, axis=0)
    img2_in_ar = np.expand_dims(img2_in_ar, axis=0)

    loss_psnr = skimage.measure.compare_psnr(img1_in_ar, img2_in_ar)
    total_loss += loss_psnr

  return total_loss / len(image_set1_paths)


def compute_lpips_metric(image_set1_paths, image_set2_paths, debug=False):
  assert len(image_set1_paths) == len(image_set2_paths)
  assert len(image_set1_paths) > 0
  print('Evaluating lpips distance for %d pairs' % len(image_set1_paths))

  img_size = 256  # all images will be resized to [img_size x img_size]
  xx = tf.placeholder("float", [None, img_size, img_size, 3])
  yy = tf.placeholder("float", [None, img_size, img_size, 3])
  lpips_loss = lpips.LPIPS(xx, yy)

  total_loss = 0.
  with tf.Session(config=tf.ConfigProto(
      allow_soft_placement=True,
      log_device_placement=False)) as sess:
    if debug:
      loss_vals = []
    for ii, (img1_path, img2_path) in enumerate(zip(image_set1_paths,
                                                    image_set2_paths)):
      conc_img = Image.open(img1_path)
      conc_img = conc_img.resize((img_size * 3, img_size))
      conc_img_ar = np.array(conc_img)
      img_width = conc_img_ar.shape[1]
      img1_in_ar = conc_img_ar[:, img_width//3:-img_width//3, :]
      img2_in_ar = conc_img_ar[:, -img_width//3:, :]

      img1_in_ar = img1_in_ar * 2./255. - 1
      img1_in_ar = np.expand_dims(img1_in_ar, axis=0)

      img2_in_ar = img2_in_ar * 2./255. - 1
      img2_in_ar = np.expand_dims(img2_in_ar, axis=0)

      feed_dict = {xx: img1_in_ar, yy: img2_in_ar}
      loss_val = sess.run(lpips_loss(), feed_dict=feed_dict)
      total_loss += loss_val

      if debug:
        basename = osp.basename(img1_path)
        print('lpips: %s: %.3f' % (basename, loss_val))
        loss_vals.append(loss_val)

  if debug:
    return total_loss / len(image_set1_paths), loss_vals
  return total_loss / len(image_set1_paths)


def compute_perceptual_loss_metric(image_set1_paths, image_set2_paths, debug=False):
  assert len(image_set1_paths) == len(image_set2_paths)
  assert len(image_set1_paths) > 0
  print('Evaluating perceptual loss for %d pairs' % len(image_set1_paths))

  img_size = 256
  xx = tf.placeholder("float", [None, img_size, img_size, 3])
  yy = tf.placeholder("float", [None, img_size, img_size, 3])
  vgg_layers = ['conv%d_2' % i for i in range(1, 6)]  # conv1 through conv5
  vgg_layer_weights = [1./32, 1./16, 1./8, 1./4, 1.]
  vgg_loss = losses.PerceptualLoss(xx, yy, [img_size, img_size, 3], vgg_layers,
                                   vgg_layer_weights)

  total_loss = 0.
  with tf.Session(config=tf.ConfigProto(
      allow_soft_placement=True,
      log_device_placement=False)) as sess:
    if debug:
      loss_vals = []
    for ii, (img1_path, img2_path) in enumerate(zip(image_set1_paths,
                                                    image_set2_paths)):
      conc_img = Image.open(img1_path)
      conc_img = conc_img.resize((img_size * 3, img_size))
      conc_img_ar = np.array(conc_img)
      img_width = conc_img_ar.shape[1]
      img1_in_ar = conc_img_ar[:, img_width//3:-img_width//3, :]
      img2_in_ar = conc_img_ar[:, -img_width//3:, :]

      img1_in_ar = img1_in_ar * 2./255. - 1
      img1_in_ar = np.expand_dims(img1_in_ar, axis=0)

      img2_in_ar = img2_in_ar * 2./255. - 1
      img2_in_ar = np.expand_dims(img2_in_ar, axis=0)

      feed_dict = {xx: img1_in_ar, yy: img2_in_ar}
      loss_val = sess.run(vgg_loss(), feed_dict=feed_dict)
      total_loss += loss_val

      if debug:
        basename = osp.basename(img1_path)
        print('lpips: %s: %.3f' % (basename, loss_val))
        loss_vals.append(loss_val)

  return total_loss / len(image_set1_paths)


def eval_metrics(results_dir, subset, datasets, methods, metrics):
  f = open(osp.join(results_dir, 'metrics_logs.txt'), 'a')
  f.write(str(datetime.datetime.now()) + '\n')
  for metric in metrics:
    for dataset in datasets:
      for method in methods:
        val_set_dir = osp.join(results_dir, subset)
        input_pattern = osp.join(val_set_dir, '*.png')
        print('Evaluating images from %s' % input_pattern)
        conc_img_set = glob.glob(input_pattern)
        if metric == 'l1':
          mean_loss = compute_l1_loss_metric(conc_img_set, conc_img_set)
        elif metric == 'lpips':
          mean_loss = compute_lpips_metric(conc_img_set, conc_img_set)
        elif metric == 'vgg':
          mean_loss = compute_perceptual_loss_metric(conc_img_set, conc_img_set)
        elif metric == 'psnr':
          mean_loss = compute_psnr_loss_metric(conc_img_set, conc_img_set)
        print('mean %s loss for %s-%s: %s = %f' % (metric, dataset, subset,
                                                  method, mean_loss))
        f.write('mean %s loss for %s-%s: %s = %f\n' % (metric, dataset, subset,
                                                   method, mean_loss))
        f.flush()
  f.close()


def main(argv):
  metrics = ['l1', 'psnr', 'lpips', 'vgg']
  results_dir = opts.inference_output_dir
  subset = opts.subset
  datasets = [FLAGS.dataset]
  methods = [FLAGS.exp_title]
  eval_metrics(results_dir, subset, datasets, methods, metrics)


if __name__ == '__main__':
  app.run(main)
