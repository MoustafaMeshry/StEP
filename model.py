from options import FLAGS as opts
import layers
import losses
import networks
import tensorflow as tf
import utils


def create_computation_graph(
    xa_in_all, gt_a_all, d_lr, g1_lr, g2_lr, arch_type, xb_in_all=None,
    gt_b_all=None, use_appearance=True, num_gpus=1):
  """Create the models and the losses for the DRIT training pipeline.

  Args:
    TODO

  Returns:
    Dictionary of placeholders and TF graph functions.
  """

  ps_device = '/cpu:0' if num_gpus > 1 else '/gpu:0'

  with tf.device(ps_device):
  # with tf.name_scope('aragoz'):
    tower_grads_d1 = []
    tower_grads_d2 = []
    tower_grads_g1 = []
    tower_grads_g2 = []
    summaries = []

    assert opts.batch_size % num_gpus == 0, 'batch size not divisble by num_gpus!'
    batch_size = opts.batch_size // num_gpus

    opt_d1 = tf.train.AdamOptimizer(d_lr, opts.adam_beta1, opts.adam_beta2,
                                    name='Adam_D1')
    opt_g1 = tf.train.AdamOptimizer(g1_lr, opts.adam_beta1, opts.adam_beta2,
                                    name='Adam_G1')
    if not opts.use_single_disc:
      opt_d2 = tf.train.AdamOptimizer(d_lr, opts.adam_beta1, opts.adam_beta2,
                                      name='Adam_D2')
    use_g2_optimizer = opts.loss_D_rand_z or opts.loss_z_rand_recon or opts.loss_z_recon
    if use_g2_optimizer:
      opt_g2 = tf.train.AdamOptimizer(g2_lr, opts.adam_beta1, opts.adam_beta2,
                                      name='Adam_G2')
    # Loop over all GPUs and construct their own computation graph
    for gpu_id in range(num_gpus):
      with tf.device(utils.assign_to_device('/gpu:{}'.format(gpu_id),
                                            ps_device=ps_device)):
        with tf.name_scope('tower_%d' % gpu_id):
          # Split data between GPUs
          xa_in = xa_in_all[gpu_id * batch_size: (gpu_id+1) * batch_size]
          gt_a = gt_a_all[gpu_id * batch_size: (gpu_id+1) * batch_size]
          if xb_in_all is not None:
            xb_in = xb_in_all[gpu_id * batch_size: (gpu_id+1) * batch_size]
            gt_b = gt_b_all[gpu_id * batch_size: (gpu_id+1) * batch_size]
          else:
            xb_in, gt_b = None, None
  
          tower = create_tower(
              xa_in, gt_a, xb_in, gt_b, arch_type, reuse=gpu_id>0,
              use_appearance=use_appearance, tb_vis_scalars=gpu_id<2,
              tb_vis_images=gpu_id==0)

          summaries += tower['tower_summaries']

          # ---------------------------------------------------------------------------
          # Optimizers
          # ---------------------------------------------------------------------------

          # D optimizer
          d1_vars = utils.model_vars('d_model_1')[0]
          train_d1_grads = opt_d1.compute_gradients(tower['loss_d'], d1_vars)
          tower_grads_d1.append(train_d1_grads)

          # D2 optimizer
          if not opts.use_single_disc:
            d2_vars = utils.model_vars('d_model_2')[0]
            train_d2_grads = opt_d2.compute_gradients(tower['loss_d2'], d2_vars)
            tower_grads_d2.append(train_d2_grads)

          # G_and_E optimizer: includes E only if opts.train_app_encoder is true
          G_and_E_vars = utils.model_vars('g_model')[0]
          E_only_vars = utils.model_vars('appearance_net')[0]
          if opts.train_app_encoder:
            G_and_E_vars += E_only_vars
          train_g_grads = opt_g1.compute_gradients(tower['loss_g'], G_and_E_vars)
          tower_grads_g1.append(train_g_grads)

          # G_only optimizer
          G_only_vars = utils.model_vars('g_model')[0]
          if use_g2_optimizer:
            train_g2_grads = opt_g2.compute_gradients(tower['loss_g2'], G_only_vars)
            if opts.train_app_encoder and tower['loss_g2_shadow'] is not None:
              train_g2_grads_E = opt_g2.compute_gradients(tower['loss_g2_shadow'], E_only_vars)
              tower_grads_g2.append(train_g2_grads + train_g2_grads_E)
            else:
              tower_grads_g2.append(train_g2_grads)

    avg_grads_d1 = utils.average_gradients(tower_grads_d1)
    train_d1 = opt_d1.apply_gradients(avg_grads_d1)
    if not opts.use_single_disc:
      avg_grads_d2 = utils.average_gradients(tower_grads_d2)
      train_d2 = opt_d2.apply_gradients(avg_grads_d2)
    else:
      train_d2 = []
    avg_grads_g1 = utils.average_gradients(tower_grads_g1)
    train_g1 = opt_g1.apply_gradients(avg_grads_g1)
    train_g_all = [train_g1]
    if use_g2_optimizer:
      avg_grads_g2 = utils.average_gradients(tower_grads_g2)
      train_g2 = opt_g2.apply_gradients(avg_grads_g2)
      train_g_all += [train_g2]
    else:
      train_g2 = []

    ema = tf.train.ExponentialMovingAverage(decay=0.999)
    with tf.control_dependencies(train_g_all):
      inference_vars_all = G_only_vars
      inference_vars_all += E_only_vars
      ema_op = ema.apply(inference_vars_all)

  summary_op = tf.summary.merge(summaries)

  print('\n\n***************************************************')
  print('len(inference_vars_all) = %d' % len(inference_vars_all))
  for ii, v in enumerate(inference_vars_all):
    print('%03d) %s' % (ii, str(v)))
  print('-------------------------------------------------------')
  print('len(d1_vars) = %d' % len(d1_vars))
  for ii, v in enumerate(d1_vars):
    print('%03d) %s' % (ii, str(v)))
  if not opts.use_single_disc:
    print('-------------------------------------------------------')
    print('len(d2_vars) = %d' % len(d2_vars))
    for ii, v in enumerate(d2_vars):
      print('%03d) %s' % (ii, str(v)))
  print('***************************************************\n\n')

  op_dict = {
      'train_disc1_op': train_d1,
      'train_disc2_op': train_d2,
      'train_g_op': ema_op,
      'summary_op': summary_op}
  # Merge op_dict with tower dict
  ret_dict = op_dict
  ret_dict.update(tower)  # just return losses from one tower
  return ret_dict


def create_tower(
    xa_in, gt_a, xb_in=None, gt_b=None, arch_type='pix2pixhd',
    reuse=tf.AUTO_REUSE, use_appearance=True, tb_vis_scalars=True,
    tb_vis_images=True):

  # prepare appearance/style input
  xa_app_in = gt_a
  xb_app_in = gt_b
  use_cross_cycle = xb_in is not None

  # --------------------------------------------------------------------
  # Build models/networks
  # --------------------------------------------------------------------
  assert arch_type == 'pix2pixhd'
  cond_generator = networks.MultiModalConditionalGANModel(
      arch_type, use_appearance, reuse=reuse)
  E_content = cond_generator.get_content_encoder()
  G = cond_generator.get_generator()
  E_app = cond_generator.get_appearance_encoder()

  disc_class = networks.MultiScaleDiscriminator
  disc1 = disc_class(
      'd_model_1', opts.input_nc + opts.output_nc, num_scales=opts.num_d_scales,
      nf=opts.d_nf, n_layers=opts.num_d_layers, get_fmaps=False, reuse=reuse)
  if opts.use_single_disc:
    disc2 = disc1
  else:
    disc2 = disc_class('d_model_2', opts.input_nc + opts.output_nc,
                       num_scales=opts.num_d_scales, nf=opts.d_nf,
                       n_layers=opts.num_d_layers, get_fmaps=False, reuse=reuse)

  # -------------------------------------------------------------------
  # Forward pass
  # -------------------------------------------------------------------

  def get_random_z(batch_size=opts.batch_size, mean=0, std=1):
    return tf.random_normal(shape=[batch_size, 1, 1, opts.app_vector_size],
                            mean=mean, stddev=std, dtype=tf.float32)

  # Compute the latent appearance vector for images A, B
  z_a, mean_a, logvar_a = E_app(xa_app_in)
  if use_cross_cycle:
    z_b, mean_b, logvar_b = E_app(xb_app_in)

  if opts.use_vae and (opts.use_concat or opts.use_bicycle_gan_ez):
    eps_a = get_random_z(tf.shape(xa_in)[0], mean=0, std=1)
    z_a = mean_a + eps_a * tf.exp(0.5 * logvar_a)
    if use_cross_cycle:
      eps_b = get_random_z(tf.shape(xb_in)[0], mean=0, std=1)
      z_b = mean_b + eps_b * tf.exp(0.5 * logvar_b)

  # Compute the input (content) encoding and feature maps
  # Feature maps are needed for the skip connections to G
  if opts.model_name == 'pix2pixhd':
    a_content_fmaps = E_content(xa_in)
    a_content = a_content_fmaps[0]
    if use_cross_cycle:
      b_content_fmaps = E_content(xb_in)
      b_content = b_content_fmaps[0]

  # Direct (1-step) reconstruction of images A, B
  if opts.model_name == 'pix2pixhd':
    recon_a = G(a_content, a_content_fmaps, z_a)
    if use_cross_cycle:
      recon_b = G(b_content, b_content_fmaps, z_b)
  else:
    recon_a = G(xa_in, z_a, None)
    if use_cross_cycle:
      recon_b = G(xb_in, z_b, None)

  if opts.loss_z_recon:
    assert use_cross_cycle, 'loss_z_recon requires cross-cycle training mode'
    # za_recon, _, _ = E_app(recon_a)
    # if use_cross_cycle:
    #   zb_recon, _, _ = E_app(recon_b)

  if opts.loss_z_rand_recon or opts.loss_D_rand_z:
    # Sample a random appearance vector z
    z_sampled = get_random_z(tf.shape(xa_in)[0], mean=opts.z_mean,
                             std=opts.z_std)

    # Generate images A, B with the sampled appearance
    if opts.model_name == 'pix2pixhd':
      a_sampled_z = G(a_content, a_content_fmaps, z_sampled)
      if use_cross_cycle:
        b_sampled_z = G(b_content, b_content_fmaps, z_sampled)
    else:
      a_sampled_z = G(xa_in, z_sampled, None)
      if use_cross_cycle:
        b_sampled_z = G(xb_in, z_sampled, None)

    if opts.loss_z_rand_recon:
      # Reconstruct the sampled z vector
      z_recon_from_a, mean_a_3, _ = E_app(a_sampled_z)
      if use_cross_cycle:
        z_recon_from_b, mean_b_3, _ = E_app(b_sampled_z)
      # This is weird, but this is how it is done in BicycleGAN and DRIT
      if opts.use_vae and (opts.use_concat or opts.use_bicycle_gan_ez):
        z_recon_from_a = mean_a_3
        if use_cross_cycle:
          z_recon_from_b = mean_b_3

  # Generate images with swapped appearances
  if use_cross_cycle:
    if opts.model_name == 'pix2pixhd':
      a_app_b = G(a_content, a_content_fmaps, z_b)
      b_app_a = G(b_content, b_content_fmaps, z_a)
    else:
      a_app_b = G(xa_in, z_b, None)
      b_app_a = G(xb_in, z_a, None)

    # Cycle computation of the z vectors from the swapped appearance images
    cycle_z_a, mean_a_2, logvar_a_2 = E_app(b_app_a)
    cycle_z_b, mean_b_2, logvar_b_2 = E_app(a_app_b)
    # Create shadows for cycle z reconstructions for tf.stop_gradient()
    if opts.loss_z_recon:
      cycle_z_a_shadow = tf.stop_gradient(cycle_z_a)
      cycle_z_b_shadow = tf.stop_gradient(cycle_z_b)

    # Cycle (2-step) reconstruction of the input images A, B
    if opts.loss_cyclic_recon or opts.loss_D_cyclic:
      if opts.use_vae and (opts.use_concat or opts.use_bicycle_gan_ez):
        eps_a2 = get_random_z(tf.shape(xa_in)[0], mean=0, std=1)
        eps_b2 = get_random_z(tf.shape(xb_in)[0], mean=0, std=1)
        cycle_z_a = mean_a_2 + eps_a2 * tf.exp(0.5 * logvar_a_2)
        cycle_z_b = mean_b_2 + eps_b2 * tf.exp(0.5 * logvar_b_2)
      if opts.model_name == 'pix2pixhd':
        cycle_a = G(a_content, a_content_fmaps, cycle_z_a)
        cycle_b = G(b_content, b_content_fmaps, cycle_z_b)
      else:
        cycle_a = G(xa_in, cycle_z_a, None)
        cycle_b = G(xb_in, cycle_z_b, None)

  # -------------------------------------------------------------------
  # Compute discriminator logits
  # -------------------------------------------------------------------

  # 1. Real logits for ground truth (using both discriminators)
  disc1_real_a_fmaps = disc1(gt_a, xa_in)
  if use_cross_cycle:
    disc1_real_b_fmaps = disc1(gt_b, xb_in)
  if not opts.use_single_disc:
    disc2_real_a_fmaps = disc2(gt_a, xa_in)
    if use_cross_cycle:
      disc2_real_b_fmaps = disc2(gt_b, xb_in)
  else:
    disc2_real_a_fmaps = disc1_real_a_fmaps
    if use_cross_cycle:
      disc2_real_b_fmaps = disc1_real_b_fmaps

  # 2. Fake logits
  # 2.1 fake images from direct reconstruction
  if opts.loss_D_direct:
    disc1_fake_a_dir = disc1(recon_a, xa_in)
    if use_cross_cycle:
      disc1_fake_b_dir = disc1(recon_b, xb_in)
  # 2.2 fake A, B with a randomly sampled appearance
  if opts.loss_D_rand_z:
    disc2_fake_a_sampled_z_fmaps = disc2(a_sampled_z, xa_in)
    if use_cross_cycle:
      disc2_fake_b_sampled_z_fmaps = disc2(b_sampled_z, xb_in)
  # 2.3 fake images with swapped appearance
  if opts.loss_D_swap_z:
    disc1_fake_a_app_b_fmaps = disc1(a_app_b, xa_in)
    disc1_fake_b_app_a_fmaps = disc1(b_app_a, xb_in)
  # 2.4 fake images from cyclic reconstruction
  if opts.loss_D_cyclic:
    disc1_fake_a_cyc = disc1(cycle_a, xa_in)
    if use_cross_cycle:
      disc1_fake_b_cyc = disc1(cycle_b, xb_in)

  # -------------------------------------------------------------------
  # Losses
  # -------------------------------------------------------------------

  # Loss weights
  recon_loss_weight = opts.w_loss_vgg if opts.use_vgg_loss else opts.w_loss_l1
  w_direct_recon = recon_loss_weight
  w_cyclic_recon = recon_loss_weight
  w_gan_img = opts.w_loss_gan
  w_z_recon = opts.w_loss_z_recon
  w_z_kl = opts.w_loss_z_kl
  w_z_l2 = opts.w_loss_z_l2

  # Initialize losses
  # 1. initialize losses for different networks
  loss_g = tf.zeros([], dtype=tf.float32)  # generator loss
  loss_g2 = tf.zeros([], dtype=tf.float32)  # generator loss 2
  loss_d1 = tf.zeros([], dtype=tf.float32)  # discriminator1 loss
  loss_d2 = tf.zeros([], dtype=tf.float32)  # discriminator2 loss
  # 2. initialize loss terms
  loss_direct_recon = tf.zeros([], dtype=tf.float32)
  loss_cyclic_recon = tf.zeros([], dtype=tf.float32)
  loss_gan_direct = tf.zeros([], dtype=tf.float32)
  loss_gan_cyclic = tf.zeros([], dtype=tf.float32)
  loss_gan_swap_z = tf.zeros([], dtype=tf.float32)
  loss_gan_rand_z = tf.zeros([], dtype=tf.float32)
  loss_z_recon = tf.zeros([], dtype=tf.float32)
  loss_z_rand_recon = tf.zeros([], dtype=tf.float32)
  loss_z_kl = tf.zeros([], dtype=tf.float32)
  loss_z_l2 = tf.zeros([], dtype=tf.float32)

  # Direct reconstruction
  if opts.loss_direct_recon:
    if opts.use_vgg_loss:
      vgg_layers = ['conv%d_2' % i for i in range(1, 6)]  # conv1 through conv5
      vgg_layer_weights = [1./32, 1./16, 1./8, 1./4, 1.]
      direct_recon_a = losses.PerceptualLoss(recon_a, gt_a, [256, 256, 3], vgg_layers, vgg_layer_weights)  # TODO: don't hardcode image size!
      loss_direct_recon += direct_recon_a()
      if use_cross_cycle:
        direct_recon_b = losses.PerceptualLoss(recon_b, gt_b, [256, 256, 3], vgg_layers, vgg_layer_weights)  # TODO: don't hardcode image size!
        loss_direct_recon += direct_recon_b()
    else:
      loss_direct_recon += losses.L1_loss(recon_a, gt_a)
      if use_cross_cycle:
        loss_direct_recon += losses.L1_loss(recon_b, gt_b)

  # Sampled z reconstruction
  if opts.loss_z_rand_recon:
    loss_z_rand_recon += losses.L1_loss(z_recon_from_a, z_sampled)
    if use_cross_cycle:
      loss_z_rand_recon += losses.L1_loss(z_recon_from_b, z_sampled)

  # z reconstruction
  if opts.loss_z_recon:
    # loss_z_recon += losses.L1_loss(za_recon, z_a)
    loss_z_recon += losses.L1_loss(cycle_z_a, z_a)
    loss_z_recon_shadow = losses.L1_loss(cycle_z_a_shadow, z_a)
    if use_cross_cycle:
      # loss_z_recon += losses.L1_loss(zb_recon, z_b)
      loss_z_recon += losses.L1_loss(cycle_z_b, z_b)
      loss_z_recon_shadow += losses.L1_loss(cycle_z_b_shadow, z_b)

  # Cyclic reconstruction
  if use_cross_cycle and opts.loss_cyclic_recon:
    if opts.use_vgg_loss:
      vgg_layers = ['conv%d_2' % i for i in range(1, 6)]  # conv1 through conv5
      vgg_layer_weights = [1./32, 1./16, 1./8, 1./4, 1.]
      cyclic_recon_a = losses.PerceptualLoss(cycle_a, gt_a, [256, 256, 3], vgg_layers, vgg_layer_weights)  # TODO: don't hardcode image size!
      cyclic_recon_b = losses.PerceptualLoss(cycle_b, gt_b, [256, 256, 3], vgg_layers, vgg_layer_weights)  # TODO: don't hardcode image size!
      loss_cyclic_recon += cyclic_recon_a()
      loss_cyclic_recon += cyclic_recon_b()
    else:
      loss_cyclic_recon += losses.L1_loss(cycle_a, gt_a)
      loss_cyclic_recon += losses.L1_loss(cycle_b, gt_b)

  # GAN loss for the image encoder-generator (E, G) and discriminator (D)
  # 1. Real D loss
  loss_d_real_terms = [tf.zeros([], dtype=tf.float32)]
  loss_d_fake_terms = [tf.zeros([], dtype=tf.float32)]
  loss_d2_real_terms = [tf.zeros([], dtype=tf.float32)]
  loss_d2_fake_terms = [tf.zeros([], dtype=tf.float32)]
  loss_g_gan_terms = [tf.zeros([], dtype=tf.float32)]
  loss_g2_gan_terms = [tf.zeros([], dtype=tf.float32)]

  loss_d_real_terms.append(
      losses.multiscale_discriminator_loss(disc1_real_a_fmaps, True))
  if use_cross_cycle:
    loss_d_real_terms.append(
        losses.multiscale_discriminator_loss(disc1_real_b_fmaps, True))
  loss_d_real = tf.reduce_mean(loss_d_real_terms)
  if not opts.use_single_disc:
    loss_d2_real_terms.append(
        losses.multiscale_discriminator_loss(disc2_real_a_fmaps, True))
    if use_cross_cycle:
      loss_d2_real_terms.append(
          losses.multiscale_discriminator_loss(disc2_real_b_fmaps, True))
    loss_d2_real = tf.reduce_mean(loss_d2_real_terms)
  else:
    loss_d2_real = loss_d_real

  # 1. GAN loss on direct reconstructions
  if opts.loss_D_direct:
    loss_d_fake_terms.append(
        losses.multiscale_discriminator_loss(disc1_fake_a_dir, False))
    loss_g_gan_terms.append(
        losses.multiscale_discriminator_loss(disc1_fake_a_dir, True))
    if use_cross_cycle:
      loss_d_fake_terms.append(
          losses.multiscale_discriminator_loss(disc1_fake_b_dir, False))
      loss_g_gan_terms.append(
          losses.multiscale_discriminator_loss(disc1_fake_b_dir, True))

  # 2. GAN loss on synthesized images with randomly sampled z
  if opts.loss_D_rand_z:  # could be using a separate discriminator!
    loss_g2_gan_terms.append(
        losses.multiscale_discriminator_loss(disc2_fake_a_sampled_z_fmaps, True))
    if use_cross_cycle:
      loss_g2_gan_terms.append(
          losses.multiscale_discriminator_loss(disc2_fake_b_sampled_z_fmaps, True))
    if opts.use_single_disc:
      loss_d_fake_terms.append(
          losses.multiscale_discriminator_loss(disc2_fake_a_sampled_z_fmaps, False))
      if use_cross_cycle:
        loss_d_fake_terms.append(
            losses.multiscale_discriminator_loss(disc2_fake_b_sampled_z_fmaps, False))
    else:
      loss_d2_fake_terms.append(
          losses.multiscale_discriminator_loss(disc2_fake_a_sampled_z_fmaps, False))
      if use_cross_cycle:
        loss_d2_fake_terms.append(
            losses.multiscale_discriminator_loss(disc2_fake_b_sampled_z_fmaps, False))

  # 3. GAN loss on synthesized images with swapped zs
  if use_cross_cycle and opts.loss_D_swap_z:
    loss_d_fake_terms.append(
        losses.multiscale_discriminator_loss(disc1_fake_a_app_b_fmaps, False))
    loss_d_fake_terms.append(
        losses.multiscale_discriminator_loss(disc1_fake_b_app_a_fmaps, False))
    if opts.loss_z_recon:
      loss_g2_gan_terms.append(
          losses.multiscale_discriminator_loss(disc1_fake_a_app_b_fmaps, True))
      loss_g2_gan_terms.append(
          losses.multiscale_discriminator_loss(disc1_fake_b_app_a_fmaps, True))
    else:
      loss_g_gan_terms.append(
          losses.multiscale_discriminator_loss(disc1_fake_a_app_b_fmaps, True))
      loss_g_gan_terms.append(
          losses.multiscale_discriminator_loss(disc1_fake_b_app_a_fmaps, True))

  # 4. GAN loss on cyclic reconstructions
  if use_cross_cycle and opts.loss_D_cyclic:
    loss_d_fake_terms.append(
        losses.multiscale_discriminator_loss(disc1_fake_a_cyc, False))
    loss_d_fake_terms.append(
        losses.multiscale_discriminator_loss(disc1_fake_b_cyc, False))
    loss_g_gan_terms.append(
        losses.multiscale_discriminator_loss(disc1_fake_a_cyc, True))
    loss_g_gan_terms.append(
        losses.multiscale_discriminator_loss(disc1_fake_b_cyc, True))

  # Aggregate GAN loss terms
  # disc1 loss
  loss_d_real = tf.reduce_mean(loss_d_real_terms)
  loss_d_fake = tf.reduce_mean(loss_d_fake_terms)
  loss_d_gan = loss_d_real + loss_d_fake

  # disc2 loss
  if opts.use_single_disc:
    loss_d2_gan = tf.zeros([], dtype=tf.float32)
    loss_d2_real = tf.zeros([], dtype=tf.float32)
    loss_d2_fake = tf.zeros([], dtype=tf.float32)
  else:
    loss_d2_real = tf.reduce_mean(loss_d2_real_terms)
    loss_d2_fake = tf.reduce_mean(loss_d2_fake_terms)
    loss_d2_gan = loss_d2_real + loss_d2_fake

  # GAN loss for G+E optimizer
  loss_g_gan = tf.reduce_mean(loss_g_gan_terms)

  # GAN loss for G-only optimizer
  loss_g2_gan = tf.reduce_mean(loss_g2_gan_terms)

  # KL divergence
  if opts.loss_z_kl:
    loss_z_kl = losses.KL_loss(mean_a, logvar_a)
    if use_cross_cycle:
      loss_z_kl += losses.KL_loss(mean_b, logvar_b)

  if opts.loss_z_l2:
    loss_z_l2 = losses.l2_regularize(z_a)
    if use_cross_cycle:
      loss_z_l2 += losses.l2_regularize(z_b)

  # G+E loss (E is included only if trainable)
  loss_g_terms = []
  if opts.loss_direct_recon:
    loss_g_terms.append(w_direct_recon * loss_direct_recon)
  if opts.loss_cyclic_recon:
    loss_g_terms.append(w_cyclic_recon * loss_cyclic_recon)
  if opts.loss_g_gan:
    loss_g_terms.append(w_gan_img * loss_g_gan)
  if opts.loss_z_l2:
    loss_g_terms.append(w_z_l2 * loss_z_l2)
  if opts.loss_z_kl and (opts.use_bicycle_gan_ez or opts.use_concat):
    loss_g_terms.append(w_z_kl * loss_z_kl)
  if len(loss_g_terms) > 0:
    loss_g = sum(loss_g_terms)

  # G_only loss
  loss_g2_terms = []
  if opts.loss_D_rand_z or (opts.loss_D_swap_z and opts.loss_z_recon):
    loss_g2_terms.append(w_gan_img * loss_g2_gan)  # could be using a separate discriminator!
  if opts.loss_z_rand_recon:
    loss_g2_terms.append(w_z_recon * loss_z_rand_recon)
  if opts.loss_z_recon:
    # Add loss_z_recon to opt_G_only, and it'll update G only once.
    loss_g2_terms.append(w_z_recon * loss_z_recon)
  if len(loss_g2_terms) > 0:
    loss_g2 = sum(loss_g2_terms)

  # Discriminator(s) loss
  loss_d1 = loss_d_gan
  loss_d2 = loss_d2_gan

  # -------------------------------------------------------------------
  # Tensorboard visualizations
  # -------------------------------------------------------------------

  tower_summaries = []
  # Image summaries
  if tb_vis_images:
    tb_vis_a = tf.concat([xa_in, gt_a, recon_a], axis=2)
    if use_cross_cycle:
      if opts.loss_cyclic_recon or opts.loss_D_cyclic:
        tb_vis_a = tf.concat([tb_vis_a, a_app_b, cycle_a], axis=2)
        tb_vis_b = tf.concat([xb_in, gt_b, recon_b, b_app_a, cycle_b], axis=2)
      else:
        tb_vis_a = tf.concat([tb_vis_a, a_app_b], axis=2)
        tb_vis_b = tf.concat([xb_in, gt_b, recon_b, b_app_a], axis=2)
    if opts.loss_D_rand_z:
      tb_vis_a = tf.concat([tb_vis_a, a_sampled_z], axis=2)
      if use_cross_cycle:
        tb_vis_b = tf.concat([tb_vis_b, b_sampled_z], axis=2)

    if use_cross_cycle:
      tb_vis = tf.concat([tb_vis_a, tb_vis_b], axis=1)
    else:
      tb_vis = tb_vis_a
    with tf.name_scope('output_vis'):
      tower_summaries.append(tf.summary.image('input-gt-dRecon-cycRecon-z_rand', tb_vis))

  # Scalar loss summaries
  if tb_vis_scalars:
    with tf.name_scope('discriminators'):
      with tf.name_scope('D'):
        tower_summaries.append(tf.summary.scalar('D_real_loss', w_gan_img * loss_d_real))
        tower_summaries.append(tf.summary.scalar('D_fake_loss', w_gan_img * loss_d_fake))
        tower_summaries.append(tf.summary.scalar('D_total_loss', loss_d1))
      if not opts.use_single_disc:
        with tf.name_scope('D2'):
          tower_summaries.append(tf.summary.scalar('D2real_loss', w_gan_img * loss_d2_real))
          tower_summaries.append(tf.summary.scalar('D2fake_loss', w_gan_img * loss_d2_fake))
          tower_summaries.append(tf.summary.scalar('D2total_loss', loss_d2))
    with tf.name_scope('Generator_Loss'):
      with tf.name_scope('G_and_E'):
        if opts.loss_g_gan:
          tower_summaries.append(tf.summary.scalar('G_GAN_loss', w_gan_img * loss_g_gan))
        if opts.loss_direct_recon:
          tower_summaries.append(tf.summary.scalar('direct_recon_loss', w_direct_recon * loss_direct_recon))
        if opts.loss_cyclic_recon:
          tower_summaries.append(tf.summary.scalar('cyclic_recon_loss', w_cyclic_recon * loss_cyclic_recon))
      with tf.name_scope('G_only'):
        if opts.loss_z_recon:
          tower_summaries.append(tf.summary.scalar('z_recon_loss', w_z_recon * loss_z_recon))
        if opts.loss_z_rand_recon:
          tower_summaries.append(tf.summary.scalar('z_rand_recon_loss', w_z_recon * loss_z_rand_recon))
        if opts.loss_D_rand_z or (opts.loss_z_recon and opts.loss_D_swap_z):
          tower_summaries.append(tf.summary.scalar('G2_GAN_loss', w_gan_img * loss_g2_gan))
        if opts.loss_z_kl:
          tower_summaries.append(tf.summary.scalar('z_kl', w_z_kl * loss_z_kl))
        if opts.loss_z_l2:
          tower_summaries.append(tf.summary.scalar('z_l2', w_z_l2 * loss_z_l2))
      tower_summaries.append(tf.summary.scalar('G_total_loss', loss_g))
      tower_summaries.append(tf.summary.scalar('G2_total_loss', loss_g2))

  return {'loss_d': loss_d1,
          'loss_d2': loss_d2,
          'loss_g': loss_g,
          'loss_g2': loss_g2,
          'loss_g2_shadow': None,
          'loss_d_real': loss_d_real,
          'loss_d_fake': loss_d_fake,
          'loss_d2_real': loss_d2_real,
          'loss_d2_fake': loss_d2_fake,
          'loss_direct_recon': w_direct_recon * loss_direct_recon,
          'loss_cyclic_recon': w_cyclic_recon * loss_cyclic_recon,
          'loss_g_gan': w_gan_img * loss_g_gan,
          'loss_g2_gan': w_gan_img * loss_g2_gan,
          'loss_z_recon': w_z_recon * loss_z_recon,
          'loss_z_rand_recon': w_z_recon * loss_z_rand_recon,
          'loss_z_kl': w_z_kl * loss_z_kl,
          'loss_z_l2': w_z_l2 * loss_z_l2,
          'tower_summaries': tower_summaries}
