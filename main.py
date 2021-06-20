from PIL import Image
from absl import app
import datetime
import functools
import glob
import numpy as np
import os.path as osp
import random
import skimage.measure
import tensorflow as tf
import time

from options import FLAGS as opts
import data
import eval_metrics
import losses
import lpips
import model
import networks
import options
import utils


# This is a training function with manual multi-gpu management!
def train(dataset_name, dataset_parent_dir, load_pretrained_app_encoder,
          load_trained_fixed_app):
  image_dir = osp.join(opts.train_dir, 'images')  # to save validation images.
  tf.gfile.MakeDirs(image_dir)
  model_dir = opts.train_dir

  input_iter = data.provide_data_iter(
      dataset_name, dataset_parent_dir, batch_size=opts.batch_size,
      subset=opts.subset, repeat_flag=True, shuffle=128, data_aug=True,
      crop_size=opts.crop_size)

  # Place all ops on CPU by default
  with tf.device('/cpu:0'):
    if load_trained_fixed_app:
      global_step_init = utils.load_global_step_from_checkpoint_dir(
          opts.fixed_appearance_train_dir)
    else:
      global_step_init = 0
    global_step = tf.get_variable(tf.GraphKeys.GLOBAL_STEP, dtype=tf.int64,
                                  initializer=np.int64(global_step_init))

    # Get a batch for each GPU
    input_dict = input_iter.get_next()
    use_cross_cycle = opts.loss_cyclic_recon or opts.loss_D_cyclic or opts.loss_D_swap_z or opts.loss_z_recon
    xa_in = input_dict['conditional_input']
    gt_a = input_dict['expected_output']  # ground truth output
    if use_cross_cycle:
      xb_in = input_dict['conditional_input_2']
      gt_b = input_dict['expected_output_2']  # ground truth output
    else:
      xb_in = None
      gt_b = None

    # -------------------------------------------------------------------------
    # Build model
    # -------------------------------------------------------------------------
    d_lr_ph = tf.placeholder(tf.float32, shape=[])
    g1_lr_ph = tf.placeholder(tf.float32, shape=[])
    g2_lr_ph = tf.placeholder(tf.float32, shape=[])
    ops = model.create_computation_graph(
        xa_in, gt_a, d_lr_ph, g1_lr_ph, g2_lr_ph, opts.model_name, xb_in, gt_b,
        num_gpus=opts.num_gpus)
    op_increment_step = tf.assign_add(global_step, 1)

    train_d1_op = ops['train_disc1_op']
    train_d2_op = ops['train_disc2_op']
    train_g_op = ops['train_g_op']  # ema that includes all G and/or E optimizers
    summary_op = ops['summary_op']

    loss_names = []
    loss_ops = []
    # main loss terms for optimizers
    loss_ops.append(ops['loss_d'])
    loss_names.append('loss_d')
    if not opts.use_single_disc:
      loss_ops.append(ops['loss_d2'])
      loss_names.append('loss_d2')
    loss_ops.append(ops['loss_g'])
    loss_names.append('loss_g')
    loss_ops.append(ops['loss_g2'])
    loss_names.append('loss_g2')
    # details for individual loss terms
    loss_ops.append(ops['loss_d_real'])
    loss_names.append('loss_d_real')
    loss_ops.append(ops['loss_d_fake'])
    loss_names.append('loss_d_fake')
    loss_ops.append(ops['loss_g_gan'])
    loss_names.append('loss_g_gan')
    if opts.loss_D_rand_z:
      loss_ops.append(ops['loss_g2_gan'])
      loss_names.append('loss_g2_gan')
    if not opts.use_single_disc:
      loss_ops.append(ops['loss_d2_real'])
      loss_names.append('loss_d2_real')
      loss_ops.append(ops['loss_d2_fake'])
      loss_names.append('loss_d2_fake')
    if opts.loss_direct_recon:
      loss_ops.append(ops['loss_direct_recon'])
      loss_names.append('loss_direct_recon')
    if opts.loss_cyclic_recon:
      loss_ops.append(ops['loss_cyclic_recon'])
      loss_names.append('loss_cyclic_recon')
    if opts.loss_z_recon:
      loss_ops.append(ops['loss_z_recon'])
      loss_names.append('loss_z_recon')
    if opts.loss_z_rand_recon:
      loss_ops.append(ops['loss_z_rand_recon'])
      loss_names.append('loss_z_rand_recon')
    if opts.loss_z_kl:
      loss_ops.append(ops['loss_z_kl'])
      loss_names.append('loss_z_kl')
    if opts.loss_z_l2:
      loss_ops.append(ops['loss_z_l2'])
      loss_names.append('loss_z_l2')

    train_op = tf.group(train_d1_op, train_d2_op, train_g_op, op_increment_step)
    # </endif for building the training model>

    # Build inference/evaluation model
    with tf.name_scope('inference'):
      cond_generator = networks.MultiModalConditionalGANModel(
          opts.model_name, use_appearance=True, reuse=True)
      if opts.dataset_name in ['maps', 'space_needle_timelapse']:
        val_img_size = 512  # for valset not trainset
      else:
        val_img_size = 256
      with tf.device('/gpu:0'):
        x_eval = tf.placeholder("float", [None, val_img_size, val_img_size, 3])
        y_eval = tf.placeholder("float", [None, val_img_size, val_img_size, 3])
        lpips_inst = lpips.LPIPS(x_eval, y_eval)
        lpips_eval = lpips_inst()  # validation set loss
        vgg_layers = ['conv%d_2' % i for i in range(1, 6)]  # conv1 through conv5
        vgg_layer_weights = [1./32, 1./16, 1./8, 1./4, 1.]
        vgg_inst = losses.PerceptualLoss(x_eval, y_eval, [256, 256, 3], vgg_layers, vgg_layer_weights)  # TODO: don't hardcode image size!
        vgg_eval = vgg_inst()
        eval_ops = [lpips_eval, vgg_eval]

    # -------------------------------------------------------------------------
    # TF Session:
    # -------------------------------------------------------------------------

    # Start running operations on the Graph. allow_soft_placement MUST be set to
    # True to build towers on GPU, as some of the ops do not have GPU
    # implementations.
    with tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False)) as sess:

      # Create a saver and a summary writer
      saver = tf.train.Saver(tf.global_variables(),
                             max_to_keep=opts.n_ckpt_to_keep)
      summary_writer = tf.summary.FileWriter(opts.train_dir, sess.graph)

      # -----------------------------------------------------------------------
      # Load and/or initialize variables
      # -----------------------------------------------------------------------

      init_op = tf.global_variables_initializer()

      if tf.train.latest_checkpoint(model_dir):  # resume training
        tf.logging.warning('***** Resuming training from %s!' % model_dir)
        latest_ckpt_path = tf.train.latest_checkpoint(model_dir)
        saver.restore(sess, latest_ckpt_path)
      elif load_pretrained_app_encoder:
        tf.logging.warning('***** Attempting to warm-start from %s!' %
                           opts.appearance_pretrain_dir)
        latest_ckpt_path = tf.train.latest_checkpoint(opts.appearance_pretrain_dir)
        assert latest_ckpt_path is not None, (
            'No ckpt found at %s' % opts.appearance_pretrain_dir)
        sess.run(init_op)  # initialize the Generator and Discriminators.
        E_vars = utils.model_vars('appearance_net')[0]
        # Restore appearance encoder vars
        E_saver = tf.train.Saver(E_vars)
        E_saver.restore(sess, latest_ckpt_path)
      elif load_trained_fixed_app:
        tf.logging.warning('****** finetuning will warm-start from %s!' %
                           opts.fixed_appearance_train_dir)
        latest_ckpt_path = tf.train.latest_checkpoint(opts.fixed_appearance_train_dir)
        assert latest_ckpt_path is not None, 'No ckpt found at %s' % opts.fixed_appearance_train_dir
        sess.run(init_op)
        train_vars_saver = tf.train.Saver(tf.trainable_variables())
        train_vars_saver.restore(sess, latest_ckpt_path)
      else:
        tf.logging.warning('****** No warm-starting; using random initialization!')
        sess.run(init_op)

      # -----------------------------------------------------------------------
      # Training loop
      # -----------------------------------------------------------------------

      print('Training started at %s' % str(datetime.datetime.now()))
      t_start = time.time()
      log_steps = opts.log_steps
      summary_steps = opts.summary_steps
      ckpt_steps = opts.ckpt_steps
      fixed_lr_kimg = opts.fixed_lr_kimg  # 600
      total_kimg = opts.total_kimg  # 900
      eval_kimgs = 10
      eval_steps = (eval_kimgs * 1000) // opts.batch_size
      curr_step = 0
      decay_learning_rates = total_kimg > fixed_lr_kimg
      val_logs_filepath = osp.join(model_dir, 'val_logs.txt')

      while True:
        curr_kimg = curr_step * opts.batch_size // 1000
        if curr_kimg >= total_kimg:
          print('Training complete!')
          # Write a final summary
          summary_str = sess.run(summary_op)
          summary_writer.add_summary(summary_str, curr_step)
          # Save the last checkpoint
          checkpoint_path = osp.join(opts.train_dir, 'model.ckpt')
          print('Saving final checkpoint %s' % checkpoint_path)
          saver.save(sess, checkpoint_path, global_step=int(curr_step))
          print('final checkpoint saved!')
          batch_size=16
          num_batches=4
          if opts.dataset_name in ['maps', 'space_needle_timelapse']:
            crop_size = 512
            max_examples=128
          else:
            crop_size = 256
            max_examples = None
          # Evaluate last checkpoint
          evaluate(sess, cond_generator, eval_ops, x_eval, y_eval, curr_step,
                   curr_kimg, val_logs_filepath, dataset_name, dataset_parent_dir,
                   batch_size=batch_size,  max_examples=max_examples,
                   crop_size=crop_size)
          break

        # Set up learning rates for this iteration.
        d_lr = opts.d_lr
        g_lr = opts.g_lr
        ez_lr = opts.ez_lr
        if decay_learning_rates and curr_kimg > fixed_lr_kimg:
          decay_factor = 1 - (curr_kimg - fixed_lr_kimg) * 1. / (
              total_kimg - fixed_lr_kimg + 10)
          d_lr *= decay_factor
          g_lr *= decay_factor
          ez_lr *= decay_factor

        if opts.training_pipeline == 'staged':
          feed_dict={d_lr_ph: d_lr, g_lr_ph: g_lr, ez_lr_ph: ez_lr}
        elif opts.training_pipeline == 'custom':
          feed_dict={d_lr_ph: d_lr, g1_lr_ph: g_lr, g2_lr_ph: ez_lr}

        iter_results = sess.run([global_step, train_op] + loss_ops,
                                feed_dict=feed_dict)
        curr_step = iter_results[0]
        curr_step = int(curr_step)
        loss_vals = iter_results[2:]

        if curr_step % log_steps == 0:
          t_end = time.time()
          time_per_kimg = (t_end - t_start) * 1000 / log_steps / opts.batch_size
          time_per_step = (t_end - t_start) / log_steps
          print('global_step=%d (time/kimg = %.3f, time/step = %.3f seconds), '
                'kimgs=%d/%d' % (curr_step, time_per_kimg, time_per_step,
                                 curr_step * opts.batch_size // 1000, total_kimg))
          loss_str = ''
          for loss_name, loss_val in zip(loss_names, loss_vals):
            if len(loss_str) > 0:
              loss_str += ', '
            loss_str += '%s=%.4f' % (loss_name, loss_val)
          print('%s\n' % loss_str)
          t_start = time.time()  # restart the timer

        if curr_step % summary_steps == 0:
          summary_str = sess.run(summary_op)
          summary_writer.add_summary(summary_str, curr_step)

        if curr_step % ckpt_steps == 0:
          checkpoint_path = osp.join(opts.train_dir, 'model.ckpt')
          print('Saving checkpoint %s' % checkpoint_path)
          saver.save(sess, checkpoint_path, global_step=int(curr_step))
          print('checkpoint saved!')

        if curr_step % eval_steps == 0:
          batch_size=16
          num_batches=4
          if opts.dataset_name in ['maps', 'space_needle_timelapse']:
            crop_size = 512
            max_examples=128
          else:
            crop_size = 256
            max_examples = None
          evaluate(sess, cond_generator, eval_ops, x_eval, y_eval, curr_step,
                   curr_kimg, val_logs_filepath, dataset_name, dataset_parent_dir,
                   batch_size=batch_size, max_examples=max_examples,
                   crop_size=crop_size)

      print("Training has finished!")


def evaluate(sess, cond_generator, eval_ops, x_eval_ph, y_eval_ph, step, kimg,
              val_logs_filepath, dataset_name, dataset_parent_dir,
              batch_size=16, max_examples=None, subset='val', crop_size=256):
  """Evaluate model using val/test subsets."""
  t_start = time.time()
  input_iter = data.provide_data_iter(
      dataset_name, dataset_parent_dir, batch_size=batch_size,
      subset=subset, repeat_flag=False, shuffle=0, max_examples=max_examples,
      data_aug=False, crop_size=crop_size)
  input_dict = input_iter.get_next()
  x_in_op = input_dict['conditional_input']
  x_gt_op = input_dict['expected_output']  # ground truth output

  with tf.device('/gpu:0'):
    app_enc = cond_generator.get_appearance_encoder()
    z_app, _, _ = app_enc(x_gt_op)
    y_op = cond_generator(x_in_op, z_app)

  total_lpips_loss = 0.
  total_vgg_loss = 0.
  total_num_imgs = 0
  save_grid_flag = True
  while True:
    try:
      x_gt, y = sess.run([x_gt_op, y_op])
    except tf.errors.OutOfRangeError:
      break
    feed_dict={x_eval_ph: x_gt, y_eval_ph: y}
    lpips_loss, vgg_loss = sess.run(eval_ops, feed_dict=feed_dict)
    curr_batch_sz = x_gt.shape[0]
    total_num_imgs += curr_batch_sz
    total_lpips_loss += lpips_loss * curr_batch_sz
    total_vgg_loss += vgg_loss * curr_batch_sz

    if save_grid_flag:
      save_grid_flag = False
      _, H, W, _ = y.shape
      grid_dim = 3
      grid = np.zeros((grid_dim * H, grid_dim * W, opts.output_nc))
      idx = 0
      for i in range(grid_dim):
        for j in range(grid_dim):
          grid[i * H : (i + 1) * H, j * W: (j + 1) * W, :] = y[idx, :, :, :]
          idx += 1
      # Denormalize images:
      grid = (grid + 1) * 255. / 2
      grid = grid.astype('uint8')
      vis_img_path = osp.join(opts.train_dir, 'images', '%05d.png' % step)
      Image.fromarray(grid).save(vis_img_path)

  total_lpips_loss /= total_num_imgs
  total_vgg_loss /= total_num_imgs
  val_str = ('step %05d - %04d kimg: mean perceptual loss for %d val images = '
      '(%.4f, %.4f)' % (step, kimg, total_num_imgs, total_lpips_loss, total_vgg_loss))
  print('Validation results: ' + val_str)
  with open(val_logs_filepath, 'a+') as f:
    f.write(val_str + '\n')

  eval_time = time.time() - t_start
  print('evaluation for %d images took %.2f seconds' % (total_num_imgs, eval_time))


def build_model_fn(use_exponential_moving_average=True):
  """Builds and returns the model function for an estimator.

  Args:
    use_exponential_moving_average: bool. If true, the exponential moving
    average will be used.

  Returns:
    function, the model_fn function typically required by an estimator.
  """
  arch_type = opts.model_name
  use_appearance = True
  def model_fn(features, labels, mode, params):
    """An estimator build_fn."""
    del labels, params
    # All below modes are for different inference tasks.
    # Build network and initialize inference variables.
    g_func = networks.MultiModalConditionalGANModel(arch_type, use_appearance)
    if use_appearance:
      app_func = g_func.get_appearance_encoder()
    if use_exponential_moving_average:
      ema = tf.train.ExponentialMovingAverage(decay=0.999)
      var_dict = ema.variables_to_restore()
      tf.train.init_from_checkpoint(osp.join(opts.train_dir), var_dict)

    if mode == tf.estimator.ModeKeys.PREDICT:
      x_in = features['conditional_input']
      if use_appearance:
        x_app = features['expected_output']
        x_app_embedding, _, _ = app_func(x_app)
      else:
        x_app_embedding = None
      y = g_func(x_in, x_app_embedding)
      tf.logging.info('DBG: shape of y during prediction %s.' % str(y.shape))
      return tf.estimator.EstimatorSpec(mode=mode, predictions=y)
    # `eval_subset` mode is same as PREDICT but it concatenates the input,
    # output and ground truth in a single tuple for easy visualization.
    elif mode == 'eval_subset':
      x_in = features['conditional_input']
      x_gt = features['expected_output']
      if use_appearance:
        x_app = x_gt
        x_app_embedding, _, _ = app_func(x_app)
      else:
        x_app_embedding = None
      y = g_func(x_in, x_app_embedding)
      tf.logging.info('DBG: shape of y during prediction %s.' % str(y.shape))
      output_tuple = tf.concat([x_in, y, x_gt], axis=2)
      return tf.estimator.EstimatorSpec(mode=mode, predictions=output_tuple)
    # `compute_appearance` mode computes and returns the latent z vector.
    elif mode == 'compute_appearance':
      assert use_appearance, 'use_appearance is set to False!'
      x_app_in = features['expected_output']
      app_embedding, _, _ = app_func(x_app_in)
      return tf.estimator.EstimatorSpec(mode=mode, predictions=app_embedding)
    else:
      raise ValueError('Unsupported mode: ' + mode)

  return model_fn


def evaluate_image_set(train_dir, dataset_name, dataset_parent_dir, subset,
                       results_parent_dir, batch_size=6):
  """Runs inference on a set of images (e.g. val/test sets)."""
  experiment_title = osp.split(train_dir)[-1]
  output_dir = osp.join(results_parent_dir, subset)
  tf.gfile.MakeDirs(output_dir)
  model_fn_old = build_model_fn()
  def model_fn_wrapper(features, labels, mode, params):
    """TODO."""
    del mode
    return model_fn_old(features, labels, 'eval_subset', params)
  model_dir = train_dir
  est = tf.estimator.Estimator(model_fn_wrapper, model_dir)
  if opts.dataset_name in ['maps', 'space_needle_timelapse']:
    crop_size = 512
  else:
    crop_size = 256
  est_inp_fn = functools.partial(
      data.provide_data, dataset_name=dataset_name,
      parent_dir=dataset_parent_dir, subset=subset,
      batch_size=batch_size, repeat_flag=False, shuffle=0, crop_size=crop_size)

  print('Evaluating images for subset %s' % subset)
  images = [x for x in est.predict(est_inp_fn)]
  print('Evaluated %d images' % len(images))
  for i, img in enumerate(images):
    output_file_path = osp.join(output_dir, 'out_%04d.png' % i)
    print('Saving file #%d: %s' % (i, output_file_path))
    with tf.gfile.Open(output_file_path, 'wb') as f:
      f.write(utils.to_png(img))


def export_styles(dataset_parent_dir, dataset_name, output_dir, train_dir):
  """Saves style codes to *.npy for IMLE training of the mapper network."""
  tf.gfile.MakeDirs(output_dir)
  imgs_patterns = ['*.jpg', '*.png']
  input_dir = osp.join(dataset_parent_dir, dataset_name, 'train')
  inp_paths = sum([glob.glob(osp.join(input_dir, x)) for x in imgs_patterns], [])
  print('Number of train images = %d' % len(inp_paths))
  def input_fn():
    """TODO."""
    dict_inp = data.provide_data(
      dataset_name, dataset_parent_dir, batch_size=1, subset='train',
      repeat_flag=False, shuffle=0)
    x_gt = dict_inp['expected_output']  # ground truth output
    return {'expected_output': x_gt}

  model_fn_old = build_model_fn()
  def appearance_model_fn(features, labels, mode, params):
    """TODO."""
    del mode
    return model_fn_old(features, labels, 'compute_appearance', params)

  est_app = tf.estimator.Estimator(appearance_model_fn, train_dir)

  zs = [np.squeeze(z) for z in est_app.predict(input_fn)]
  zs = np.array(zs)
  print('Exported styles have shape of %s' % str(zs.shape))
  np.save(osp.join(output_dir, 'zs_train.npy'), zs)


def main(argv):
  del argv
  configs_str = options.list_options()
  print(configs_str)
  tf.logging.info('Local configs\n%s' % configs_str)

  if opts.run_mode == 'train':
    tf.gfile.MakeDirs(opts.train_dir)
    with tf.gfile.Open(osp.join(opts.train_dir, 'configs.txt'), 'wb') as f:
      f.write(configs_str)
    dataset_name = opts.dataset_name
    dataset_parent_dir = opts.dataset_parent_dir
    load_pretrained_app_encoder = opts.load_pretrained_app_encoder
    load_trained_fixed_app = opts.load_from_another_ckpt
    batch_size = opts.batch_size
    train(dataset_name, dataset_parent_dir, load_pretrained_app_encoder,
          load_trained_fixed_app)
    results_parent_dir = opts.train_dir
    eval_subset = 'val'
    metrics = ['l1', 'psnr', 'lpips', 'vgg']
    experiment_title = osp.basename(opts.train_dir)
    evaluate_image_set(opts.train_dir, opts.dataset_name,
                       opts.dataset_parent_dir, eval_subset, results_parent_dir,
                       opts.batch_size)
    eval_metrics.eval_metrics(results_parent_dir, eval_subset, [opts.dataset_name],
                              [experiment_title], metrics)
  elif opts.run_mode == 'eval_subset':
    subset = opts.subset
    results_parent_dir = opts.train_dir
    evaluate_image_set(opts.train_dir, opts.dataset_name,
                       opts.dataset_parent_dir, subset, results_parent_dir,
                       opts.batch_size)
  elif opts.run_mode == 'export_styles':
    output_dir = osp.join(opts.train_dir, 'exported_styles')
    dataset_parent_dir = opts.input_inference_dir
    dataset_name = opts.dataset_name
    export_styles(dataset_parent_dir, dataset_name, output_dir, opts.train_dir)
  else:
    raise ValueError('Unsupported --run_mode %s' % opts.run_mode)


if __name__ == '__main__':
  app.run(main)
