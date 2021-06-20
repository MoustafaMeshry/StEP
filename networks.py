from options import FLAGS as opts
import functools
import layers
import tensorflow as tf


class MultiModalConditionalGANModel(object):
  """TODO."""

  def __init__(self, model_name, use_appearance=True, reuse=tf.AUTO_REUSE):
    assert model_name == 'pix2pixhd', 'Model %s not implemented!' % model_name
    self._model = ModelPix2pixHD(use_appearance, reuse=reuse)

  def __call__(self, x_in, z_app=None):
    return self._model(x_in, z_app)

  def get_appearance_encoder(self):
    return self._model._appearance_encoder

  def get_generator(self):
    return self._model._generator

  def get_content_encoder(self):
    return self._model._content_encoder


# Pix2pixHD inspired model (but different from the original pix2pixHD paper)
class ModelPix2pixHD(MultiModalConditionalGANModel):
  """TODO."""

  def __init__(self, use_appearance=True, reuse=tf.AUTO_REUSE):
    self._use_appearance = use_appearance
    nf_enc = opts.g_nf
    num_downsamples = opts.p2p_n_downsamples
    num_resblocks = opts.p2p_n_resblocks
    self._content_encoder = ContentEncoderPix2pixHD(
        'g_model_enc', opts.input_nc, nf=nf_enc,
        num_downsamples=num_downsamples, num_resblocks=num_resblocks,
        reuse=reuse)
    nf_g = nf_enc * (2 ** num_downsamples)
    self._generator = GeneratorConcatV1(
        'g_model_dec', nf=nf_g, app_vec_size=opts.app_vector_size,
        conc_skip_layers=opts.concatenate_skip_layers, reuse=reuse)
    if use_appearance:
      assert opts.use_concat
      self._appearance_encoder = DRITAppearanceEncoderConcat(
          'appearance_net', opts.appearance_nc, opts.normalize_drit_Ez,
          reuse=reuse)
    else:
      self._appearance_encoder = None

  def __call__(self, x_in, z_app=None):
    x_content_fmaps = self._content_encoder(x_in)
    y = self._generator(x_content_fmaps[0], x_content_fmaps, z_app)
    return y

  def get_appearance_encoder(self):
    return self._appearance_encoder

  def get_generator(self):
    return self._generator

  def get_content_encoder(self):
    return self._content_encoder


class ContentEncoderPix2pixHD(object):
  """TODO."""

  def __init__(self, name_scope, input_nc, nf=64, num_downsamples=3,
               num_resblocks=4, reuse=tf.AUTO_REUSE):
    self.blocks = []
    self.num_downsamples = num_downsamples
    self.num_resblocks = num_resblocks
    # activation = functools.partial(tf.nn.leaky_relu, alpha=0.2)
    activation = tf.nn.relu
    # activation = functools.partial(tf.nn.leaky_relu, alpha=0.2)
    # TODO(meshry): which normalization to use? pix2pixHD uses batch_norm, PGGAN
    # uses pixel_nrom, and DRIT uses instance_norm.
    norm_layer = functools.partial(layers.LayerInstanceNorm)
    # norm_layer = functools.partial(layers.PixelNorm)
    # norm_layer = layers.pixel_norm
    # TODO(meshry): fix the bug in using padding=REFLECT and use it as in
    # pix2pixHD and DRIT
    conv2d = functools.partial(layers.LayerConv, use_scaling=opts.use_scaling,
                               relu_slope=0.2, padding='SAME')
    with tf.variable_scope(name_scope, reuse=reuse):
      # conv0: project input to nf featuremaps
      self.blocks.append(layers.LayerPipe([
          conv2d('conv0', w=7, n=[input_nc, nf], stride=1),
          norm_layer(),
          activation
      ]))

      # n_layer downsampling convolutions
      with tf.variable_scope('downsampling'):
        for ii_block in range(num_downsamples):
          with tf.variable_scope('downsampling_block%d' % ii_block):
            nf_prev = nf
            nf = min(nf * 2, 512)  # note that pix2pixHD reaches nf=1024!
            self.blocks.append(layers.LayerPipe([
                conv2d('conv%d' % (ii_block + 1), w=3, n=[nf_prev, nf],
                       stride=2),
                norm_layer(),
                activation
            ]))

      # Residual blocks
      with tf.variable_scope('res_blocks'):
        for ii_resblock in range(num_resblocks):
          # TODO(meshry): fix the bug in using padding=REFLECT and use it as in
          # pix2pixHD and DRIT
          self.blocks.append(layers.ResBlock(
              'resblock%d' % ii_resblock, nf, norm_layer, activation,
              padding='SAME', use_scaling=opts.use_scaling))

  def __call__(self, x):
    fmaps = []
    for f in self.blocks:
      x = f(x)
      fmaps.append(x)
    # fmaps.reverse()
    return fmaps[::-1]  # reverse it so that the order matches the decoder/G


# Concatentate both content fmaps + tile_and_concatenate appearance vector
class GeneratorConcatV1(object):
  """TODO."""

  def __init__(self, name_scope, nf, app_vec_size=8, num_upsamples=3,
               num_resblocks=4, conc_skip_layers=True, reuse=tf.AUTO_REUSE):
    self.app_vec_size = app_vec_size
    self.conc_skip_layers = conc_skip_layers
    self.num_resblocks = num_resblocks
    self.num_upsamples = num_upsamples
    nf_z = app_vec_size
    self.blocks = []
    activation = tf.nn.relu
    norm_layer = functools.partial(layers.LayerInstanceNorm)
    conv2d = functools.partial(layers.LayerConv, use_scaling=opts.use_scaling,
                               relu_slope=0.2)
    with tf.variable_scope(name_scope, reuse=reuse):
      # Residual blocks
      with tf.variable_scope('res_blocks'):
        for ii_resblock in range(num_resblocks):
          use_dropout = False
          self.blocks.append(layers.ResBlock(
              'resblock%d' % ii_resblock, nf + nf_z, norm_layer, activation,
              padding='SAME', use_scaling=opts.use_scaling, use_dropout=use_dropout))

      # n_layer upsampling transposed convolutions
      mult = 2 if conc_skip_layers else 1
      with tf.variable_scope('upsampling'):
        for ii_block in range(num_upsamples):
          deconv_id = num_upsamples - ii_block
          nf_prev = mult * nf
          if opts.concat_z_in_all_layers:
            nf_prev += nf_z
          if ii_block == 0:
            nf_prev += nf_z  # an extra nf_z channels preserved in resblocks.
          nf = nf // 2
          with tf.variable_scope('upscale_%d' % deconv_id):
            self.blocks.append(
                layers.LayerPipe([
                    functools.partial(layers.upscale, n=2),
                    conv2d('conv0', w=3, n=[nf_prev, nf], stride=1),
                    # norm_layer(),
                    # activation
                    activation,
                    layers.pixel_norm,
                    conv2d('conv1', w=3, n=[nf, nf], stride=1),
                    activation,
                    layers.pixel_norm
                ])
            )

        nf_prev = mult * nf
        if opts.concat_z_in_all_layers:
          nf_prev += nf_z
        self.blocks.append(layers.LayerPipe([
            conv2d('rgb', w=1, n=[nf_prev, opts.output_nc], stride=1),
            tf.tanh
        ]))

  def __call__(self, x, encoder_fmaps, app_vec):
    assert len(self.blocks) == self.num_resblocks + self.num_upsamples + 1
    if app_vec is not None:
      x = self._tile_and_concatenate(x, app_vec)
    for ii, f in enumerate(self.blocks):
      # x = f(x)
      if ii >= self.num_resblocks:
        if self.conc_skip_layers:
          x = tf.concat([encoder_fmaps[ii], x], axis=-1)
        if app_vec is not None and opts.concat_z_in_all_layers:
          x = self._tile_and_concatenate(x, app_vec)
      x = f(x)
    return x

  def _tile_and_concatenate(self, x, z):
    z = tf.reshape(z, shape=[-1, 1, 1, self.app_vec_size])
    z = tf.tile(z, [1, tf.shape(x)[1], tf.shape(x)[2], 1])
    x = tf.concat([x, z], axis=-1)
    return x


def _num_filters(fmap_base, fmap_decay, fmap_max, stage):
  # return min(int(fmap_base / (2.0**(stage * fmap_decay))), fmap_max)
  if opts.g_nf == 32:
    return min(int(2**(10 - stage)), fmap_max)  # nf32
  elif opts.g_nf == 64:
    return min(int(2**(11 - stage)), fmap_max)  # nf64
  else:
    raise ValueError('Currently unsupported num filters')


class DRITAppearanceEncoderConcat(object):
  """TODO."""

  def __init__(self, name_scope, input_nc, normalize_encoder,
               reuse=tf.AUTO_REUSE):
    self.blocks = []
    activation = functools.partial(tf.nn.leaky_relu, alpha=0.2)
    # activation = tf.nn.relu
    conv2d = functools.partial(layers.LayerConv, use_scaling=opts.use_scaling,
                               relu_slope=0.2, padding='SAME')
    with tf.variable_scope(name_scope, reuse=reuse):
      # conv0: project input to nf featuremaps
      if normalize_encoder:
        self.blocks.append(layers.LayerPipe([
            conv2d('conv0', w=4, n=[input_nc, 64], stride=2),
            layers.BasicBlock('BB0', n=[64, 128], use_scaling=opts.use_scaling),
            layers.pixel_norm,
            layers.BasicBlock('BB1', n=[128, 192], use_scaling=opts.use_scaling),
            layers.pixel_norm,
            layers.BasicBlock('BB2', n=[192, 256], use_scaling=opts.use_scaling),
            layers.pixel_norm,
            activation,
            layers.global_avg_pooling
        ]))
      else:
        self.blocks.append(layers.LayerPipe([
            conv2d('conv0', w=4, n=[input_nc, 64], stride=2),
            layers.BasicBlock('BB0', n=[64, 128], use_scaling=opts.use_scaling),
            layers.BasicBlock('BB1', n=[128, 192], use_scaling=opts.use_scaling),
            layers.BasicBlock('BB2', n=[192, 256], use_scaling=opts.use_scaling),
            activation,
            layers.global_avg_pooling
        ]))
      # FC layers to get the mean and logvar
      self.fc_mean = layers.FullyConnected(opts.app_vector_size, 'FC_mean')
      self.fc_logvar = layers.FullyConnected(opts.app_vector_size, 'FC_logvar')

  def __call__(self, x):
    for f in self.blocks:
      x = f(x)

    mean = self.fc_mean(x)
    logvar = self.fc_logvar(x)
    z = mean + tf.exp(0.5 * logvar)  # DRIT does mean + eps * stddev
    return z, mean, logvar


class PatchGANDiscriminator(object):
  """TODO."""

  def __init__(self, name_scope, input_nc, nf=64, n_layers=3, get_fmaps=False,
               use_mini_batch_stats=True, reuse=tf.AUTO_REUSE):
    """Constructor for a patchGAN discriminators.

    Args:
      name_scope: str - tf name scope.
      input_nc: int - number of input channels.
      nf: int - starting number of discriminator filters.
      n_layers: int - number of layers in the discriminator.
      get_fmaps: bool - return intermediate feature maps for FeatLoss.
    """
    self.get_fmaps = get_fmaps
    self.n_layers = n_layers
    kw = 4  # kernel width for convolution

    if opts.d_activation == 'lrelu':
      activation = functools.partial(tf.nn.leaky_relu, alpha=0.2)
    elif opts.d_activation == 'relu':
      activation = tf.nn.relu
    elif opts.d_activation == 'sigmoid':
      activation = tf.nn.sigmoid
    else:
      raise ValueError('Currently unsupported activation function for D.')
    norm_layer = functools.partial(layers.LayerInstanceNorm)
    conv2d = functools.partial(layers.LayerConv, use_scaling=opts.use_scaling,
                               relu_slope=0.2)

    def minibatch_stats(x):
      return layers.scalar_concat(x, layers.minibatch_mean_variance(x))

    # Create layers.
    self.blocks = []
    with tf.variable_scope(name_scope, reuse=reuse):
      with tf.variable_scope('block_0'):
        self.blocks.append([
            conv2d('conv0', w=kw, n=[input_nc, nf], stride=2),
            activation
        ])
      for ii_block in range(1, n_layers):
        nf_prev = nf
        nf = min(nf * 2, 512)
        with tf.variable_scope('block_%d' % ii_block):
          self.blocks.append([
              conv2d('conv%d' % ii_block, w=kw, n=[nf_prev, nf], stride=2),
              norm_layer(),
              # layers.pixel_norm,
              activation,
          ])
      # Add minibatch_stats (from PGGAN) and do a stride1 convolution.
      nf_prev = nf
      nf = min(nf * 2, 512)
      with tf.variable_scope('block_%d' % (n_layers + 1)):
        if use_mini_batch_stats:
          self.blocks.append([
              minibatch_stats,  # this is improvised by @meshry
              conv2d('conv%d' % (n_layers + 1), w=kw, n=[nf_prev + 1, nf],
                     stride=1),
              norm_layer(),
              # layers.pixel_norm,
              activation,
          ])
        else:
          self.blocks.append([
              conv2d('conv%d' % (n_layers + 1), w=kw, n=[nf_prev, nf],
                     stride=1),
              norm_layer(),
              # layers.pixel_norm,
              activation,
          ])
      # Get 1-channel patchGAN logits
      with tf.variable_scope('patchGAN_logits'):
        self.blocks.append([
            conv2d('conv%d' % (n_layers + 2), w=kw, n=[nf, 1], stride=1)
        ])

  def __call__(self, x, x_cond=None):
    # Concatenate extra conditioning input, if any.
    if x_cond is not None:
      x = tf.concat([x, x_cond], axis=3)

    if self.get_fmaps:
      # Dummy addition of x to D_fmaps, which will be removed before returing
      D_fmaps = [[x]]
      for i_block in range(len(self.blocks)):
        # Apply layer #0 in the current block
        block_fmaps = [self.blocks[i_block][0](D_fmaps[-1][-1])]
        # Apply the remaining layers of this block
        for i_layer in range(1, len(self.blocks[i_block])):
          block_fmaps.append(self.blocks[i_block][i_layer](block_fmaps[-1]))
        # Append the feature maps of this block to D_fmaps
        D_fmaps.append(block_fmaps)
      return D_fmaps[1:]  # exclude the input x which we added initially
    else:
      y = x
      for i_block in range(len(self.blocks)):
        for i_layer in range(len(self.blocks[i_block])):
          y = self.blocks[i_block][i_layer](y)
      return [[y]]


class MultiScaleDiscriminator(object):
  """TODO."""

  def __init__(self, name_scope, input_nc, num_scales=3, nf=64, n_layers=3,
               get_fmaps=False, reuse=tf.AUTO_REUSE):
    self.get_fmaps = get_fmaps
    discs = []
    with tf.variable_scope(name_scope, reuse=reuse):
      for i in range(num_scales):
        discs.append(PatchGANDiscriminator(
            'D_scale%d' % i, input_nc, nf=nf, n_layers=n_layers,
            get_fmaps=get_fmaps, reuse=reuse))
    self.discriminators = discs

  def __call__(self, x, x_cond=None, params=None):
    del params
    if x_cond is not None:
      x = tf.concat([x, x_cond], axis=3)

    responses = []
    for ii, D in enumerate(self.discriminators):
      responses.append(D(x, x_cond=None))  # x_cond is already concatenated
      if ii != len(self.discriminators) - 1:
        x = layers.downscale(x, n=2)
    return responses
