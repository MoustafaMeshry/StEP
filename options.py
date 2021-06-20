from absl import flags
import numpy as np

FLAGS = flags.FLAGS

# ------------------------------------------------------------------------------
# Train flags
# ------------------------------------------------------------------------------

# Constants and shared variables
flags.DEFINE_integer('num_gpus', 1, 'Number of training gpus.')
dataset_name = 'edges2shoes'
pipeline = 'custom'
experiment_title_suffix = None
if experiment_title_suffix:
  experiment_title = dataset_name + '_' + experiment_title_suffix
else:
  experiment_title = dataset_name
parent_train_dir = 'train'
dataset_parent_dir = 'datasets',

# Dataset, model directory and run mode
flags.DEFINE_string(
    'vgg16_path', 'third_party/pretrained_weights/vgg16.npy',
    'path to a *.npy file with vgg16 pretrained weights')
flags.DEFINE_string('pretrain_dir', '%s/app_pretrain/%s' % (
    parent_train_dir, experiment_title), 'Directory for style pretraining.')
flags.DEFINE_string('train_dir', '%s/train/%s/%s/%s' % (
    parent_train_dir, pipeline, dataset_name, experiment_title),
    'Directory for model training.')
flags.DEFINE_string('dataset_name', dataset_name, 'name ID for a dataset.')
flags.DEFINE_string('subset', 'train', '{train, val, test, ...}.')
flags.DEFINE_string('dataset_parent_dir', '%s' % dataset_parent_dir,
                    'Parent directory containing for datasets.')
flags.DEFINE_string('run_mode', 'train', "{'train', 'eval', 'infer'}")
flags.DEFINE_string(
    'imageset_dir', '%s/%s/train' % (dataset_parent_dir, dataset_name),
    'Directory containing trainset images for style pretraining.')
flags.DEFINE_integer('max_app_images', 8000, 'Max number of images to use for'
                     'style pretraining.')
flags.DEFINE_string(
    'metadata_output_dir', parent_train_dir,
    'Directory to save pairwise distance matrix for style pretraining.')
flags.DEFINE_integer('save_samples_kimg', 50, 'kimg cycle to save sample'
                     'validation ouptut during training.')
flags.DEFINE_integer('log_steps', 50, 'Frequency for logging global step, '
                     'losses, runtime after, ... etc')
flags.DEFINE_integer('summary_steps', 128,
                     'Frequency for saving tensorboard summaries.')
flags.DEFINE_integer('ckpt_steps', 1000,
                     'Frequency for saving model checkpoints.')
flags.DEFINE_integer('n_ckpt_to_keep', 10, 'Max number of checkpoints to keep.')
flags.DEFINE_integer('fixed_lr_kimg', 600, 'Number of kimgs to train with a '
                     'fixed learning rate. Learning rate will then decay '
                     'linearly between the range [fixed_lr_kimg, total_kimg].')
flags.DEFINE_integer('total_kimg', 900, 'Total number of kimgs to train with.')

# Network inputs/outputs
flags.DEFINE_boolean('use_appearance', True,
                     'Capture style from an input real image.')
flags.DEFINE_integer('input_nc', 3, 'Number of input channels.')
flags.DEFINE_integer('appearance_nc', 3,
                     'Number of input channels to the style encoder.')
flags.DEFINE_integer('output_nc', 3,
                     'Number of channels for the generated image.')
flags.DEFINE_boolean('flip_horizontal', False, 'Apply random horizontal flips')

# Staged training flags
flags.DEFINE_boolean('load_pretrained_app_encoder', False,
                     'Warmstart style encoder with pretrained weights.')
flags.DEFINE_string('appearance_pretrain_dir',
                    '%s/app_pretrain/%s/null' % (parent_train_dir, dataset_name),
                    'Model dir for the pretrained style encoder.')
flags.DEFINE_boolean('train_app_encoder', False, 'Whether to make the weights '
                     'for the style encoder trainable or not.')
flags.DEFINE_boolean(
    'load_from_another_ckpt', False, 'Load weights from another trained model, '
                     'e.g load model trained with a fixed style encoder.')
flags.DEFINE_string(
    'fixed_appearance_train_dir',
    'null' % (parent_train_dir),
    'Model dir for training G with a fixed style net.')

# -----------------------------------------------------------------------------

# More hparams
flags.DEFINE_integer('crop_size', 256, 'Crop train images to this resolution.')
flags.DEFINE_float('d_lr', 0.001, 'Learning rate for the discriminator.')
flags.DEFINE_float('g_lr', 0.001, 'Learning rate for the generator.')
flags.DEFINE_float('ez_lr', 0.0001, 'Learning rate for style encoder.')
flags.DEFINE_integer('batch_size', 8, 'Batch size for training.')
flags.DEFINE_boolean('use_scaling', True, "use He's scaling.")
flags.DEFINE_integer('num_crops', -1, 'num crops from train images'
                     '(use -1 for random crops).')
flags.DEFINE_integer('app_vector_size', 8, 'Size of latent style vector.')
flags.DEFINE_float('z_mean', 0.0, 'Mean for sampling z vectors.')
flags.DEFINE_float('z_std', 1., 'Stddev for sampling z vectors.')
flags.DEFINE_float('adam_beta1', 0.0, 'beta1 for adam optimizer.')
flags.DEFINE_float('adam_beta2', 0.99, 'beta2 for adam optimizer.')

# Losses and loss weights
flags.DEFINE_boolean('loss_g_gan', True, '')
flags.DEFINE_boolean('loss_direct_recon', True, '')
flags.DEFINE_boolean('loss_cyclic_recon', False, '')
flags.DEFINE_boolean('loss_D_direct', True, '')
flags.DEFINE_boolean('loss_D_cyclic', False, '')
flags.DEFINE_boolean('loss_D_rand_z', False, '')
flags.DEFINE_boolean('loss_D_swap_z', False, '')
flags.DEFINE_boolean('loss_z_recon', False, '')
flags.DEFINE_boolean('loss_z_rand_recon', False, '')
flags.DEFINE_boolean('loss_z_kl', False, '')
flags.DEFINE_boolean('loss_z_l2', True, '')
flags.DEFINE_boolean('use_vae', False,
                     'Use eps in KL reparamettrization trick (bicyle & DRIT).')

flags.DEFINE_float('w_loss_vgg', 0.02, 'VGG loss weight.')
flags.DEFINE_float('w_loss_feat', 2., 'Feature loss weight (from pix2pixHD).')
flags.DEFINE_float('w_loss_l1', 10., 'L1 loss weight.')
flags.DEFINE_float('w_loss_z_recon', 10, 'Z reconstruction loss weight.')
flags.DEFINE_float('w_loss_gan', 1., 'Adversarial loss weight.')
flags.DEFINE_float('w_loss_z_gan', 1., 'Z adversarial loss weight.')
flags.DEFINE_float('w_loss_z_kl', 0.01, 'KL divergence weight.')
flags.DEFINE_float('w_loss_z_l2', 0.01, 'Weight for L2 regression on Z.')

# -----------------------------------------------------------------------------

# Architecture and training setup
flags.DEFINE_string('model_name', 'pix2pixhd',
                    'Architecture type: {pggan, pix2pixhd, bicycle_gan_arch}.')
flags.DEFINE_string('training_pipeline', '%s' % pipeline,
                    'Training type type: {staged, bicycle_gan, custom, drit}.')
flags.DEFINE_integer('g_nf', 32,
                     'num filters in the first/last layers of U-net.')
flags.DEFINE_integer('d_nf', 32,
                     'num filters in the first layers of the discriminator.')
flags.DEFINE_integer('num_d_scales', 3, 'Number of multiscale discrimiantors')
flags.DEFINE_integer('num_d_layers', 3,
                     'Number of layers in the PatchGAN discriminator')
flags.DEFINE_string('d_activation', 'lrelu',
                    'activation fn for the discriminator: {relu, lrelu}.')
flags.DEFINE_boolean('use_single_disc', True,
                      'Use a single discriminator in bicycle_gan/DRIT')
flags.DEFINE_boolean('concatenate_skip_layers', True, # False,
                     'Use skip connections or not.')
flags.DEFINE_integer('pggan_n_blocks', 5,
                     'Num blocks for the pggan architecture.')
flags.DEFINE_integer('p2p_n_downsamples', 3,
                     'Num downsamples for the pix2pixHD architecture.')
flags.DEFINE_integer('p2p_n_resblocks', 4, 'Num residual blocks at the '
                     'end/start of the pix2pixHD encoder/decoder.')
flags.DEFINE_boolean('use_concat', True, '"concat" mode from DRIT.')
flags.DEFINE_boolean('use_bicycle_gan_ez', False, 'BicycleGAN Ez arch.')
flags.DEFINE_boolean('normalize_drit_Ez', True, 'Add pixelnorm layers to the '
                     'style encoder.')
flags.DEFINE_boolean('concat_z_in_all_layers', True, 'Inject z at each '
                     'upsampling layer in the decoder (only for DRIT baseline)')
flags.DEFINE_string('inject_z', 'to_encoder', 'Method for injecting z; '
                     'one of {to_encoder, to_bottleneck}.')
flags.DEFINE_boolean('use_vgg_loss', True, 'vgg v L1 reconstruction loss.')
flags.DEFINE_boolean('z_recon', False, 'Reconstruct the z vector.')
flags.DEFINE_boolean('lower_loss_g2_lrn', False, '')
flags.DEFINE_boolean('use_dropout', False, 'Only for GeneratorBicycleGAN.')

# ------------------------------------------------------------------------------
# Inference flags
# ------------------------------------------------------------------------------

flags.DEFINE_string('input_inference_dir', '',
                    'Directory containing input images for inference.')
flags.DEFINE_string('input_inference_app_dir', '',
                    'Directory containing input style images for '
                    'style transfer.')
flags.DEFINE_string('inference_output_dir', '%s/results' % parent_train_dir,
                    'Output path for inference')
flags.DEFINE_string('target_img_basename', '',
                    'basename of target image to render for interpolation')
flags.DEFINE_string('virtual_seq_name', 'full_camera_path',
                    'name for the virtual camera path suffix for the TFRecord.')
flags.DEFINE_string('inp_app_img_base_path', '',
                    'base path for the input style image for camera paths')
flags.DEFINE_string('target_img_path', '',
                    'path for the target image for interpolation')
flags.DEFINE_string('app_img1_path', '',
                    'path for the first style image for interpolation')
flags.DEFINE_string('app_img2_path', '',
                    'path for the first style image for interpolation')
flags.DEFINE_string('appearance_img1_basename', '',
                    'basename of the first style image for interpolation')
flags.DEFINE_string('appearance_img2_basename', '',
                    'basename of the first style image for interpolation')
flags.DEFINE_list('input_basenames', [], 'input basenames for inference')
flags.DEFINE_list('input_app_basenames', [], 'input style basenames for '
                  'inference')
flags.DEFINE_string('frames_dir', '',
                    'Folder with input frames to a camera path')
flags.DEFINE_string('output_dataset_name', '',
                    'dataset_name for storing results in a structured folder')
flags.DEFINE_string('input_rendered', '',
                    'input rendered image name for inference')
flags.DEFINE_string('input_depth', '', 'input depth image name for inference')
flags.DEFINE_string('input_seg', '',
                    'input segmentation mask image name for inference')
flags.DEFINE_string('input_app_rgb', '',
                    'input style rgb image name for inference')
flags.DEFINE_string('input_app_rendered', '',
                    'input style rendered image name for inference')
flags.DEFINE_string('input_app_depth', '',
                    'input style depth image name for inference')
flags.DEFINE_string('input_app_seg', '',
                    'input style segmentation mask image name for'
                    'inference')
flags.DEFINE_string('output_img_name', '',
                    '[OPTIONAL] output image name for inference')

# -----------------------------------------------------------------------------
# Some validation and assertions
# -----------------------------------------------------------------------------

def validate_options():
  if FLAGS.use_drit_training:
    assert FLAGS.use_appearance, 'DRIT pipeline requires --use_appearance'
  assert not (
    FLAGS.load_pretrained_appearance_encoder and FLAGS.load_from_another_ckpt), (
      'You cannot load weights for the style encoder from two different '
      'checkpoints!')
  if not FLAGS.use_appearance:
    print('**Warning: setting --app_vector_size to 0 since '
          '--use_appearance=False!')
    FLAGS.set_default('app_vector_size', 0)
  
# -----------------------------------------------------------------------------
# Print all options
# -----------------------------------------------------------------------------

def list_options():
  configs = ('# Run flags/options from options.py:\n'
             '# ----------------------------------\n')
  configs += ('## Train flags:\n'
              '## ------------\n')
  configs += 'vgg16_path = %s\n' % FLAGS.vgg16_path
  configs += 'pretrain_dir = %s\n' % FLAGS.pretrain_dir
  configs += 'train_dir = %s\n' % FLAGS.train_dir
  configs += 'dataset_name = %s\n' % FLAGS.dataset_name
  configs += 'subset = %s\n' % FLAGS.subset
  configs += 'dataset_parent_dir = %s\n' % FLAGS.dataset_parent_dir
  configs += 'run_mode = %s\n' % FLAGS.run_mode
  configs += 'imageset_dir = %s\n' % FLAGS.imageset_dir
  configs += 'max_app_images = %d\n' % FLAGS.max_app_images
  configs += 'metadata_output_dir = %s\n' % FLAGS.metadata_output_dir
  configs += 'save_samples_kimg = %d\n' % FLAGS.save_samples_kimg
  configs += 'log_steps = %d\n' % FLAGS.log_steps 
  configs += 'summary_steps = %d\n' % FLAGS.summary_steps 
  configs += 'ckpt_steps = %d\n' % FLAGS.ckpt_steps 
  configs += 'n_ckpt_to_keep = %d\n' % FLAGS.n_ckpt_to_keep 
  configs += 'fixed_lr_kimg = %d\n' % FLAGS.fixed_lr_kimg 
  configs += 'total_kimg = %d\n' % FLAGS.total_kimg 
  configs += '\n# --------------------------------------------------------\n\n'

  configs += ('## Network inputs and outputs:\n'
              '## ---------------------------\n')
  configs += 'use_appearance = %s\n' % str(FLAGS.use_appearance)
  configs += 'input_nc = %d\n' % FLAGS.input_nc
  configs += 'appearance_nc = %d\n' % FLAGS.appearance_nc
  configs += 'output_nc = %d\n' % FLAGS.output_nc
  configs += 'flip_horizontal = %s\n' % str(FLAGS.flip_horizontal)
  configs += 'crop_size = %d\n' % FLAGS.crop_size
  configs += '\n# --------------------------------------------------------\n\n'

  configs += ('## Staged training flags:\n'
              '## ----------------------\n')
  configs += 'load_pretrained_app_encoder = %s\n' % str(
                                            FLAGS.load_pretrained_app_encoder)
  configs += 'appearance_pretrain_dir = %s\n' % FLAGS.appearance_pretrain_dir
  configs += 'train_app_encoder = %s\n' % str(FLAGS.train_app_encoder)
  configs += 'load_from_another_ckpt = %s\n' % str(FLAGS.load_from_another_ckpt)
  configs += 'fixed_appearance_train_dir = %s\n' % str(
                                            FLAGS.fixed_appearance_train_dir)
  configs += '\n# --------------------------------------------------------\n\n'

  configs += ('## More hyper-parameters:\n'
              '## ----------------------\n')
  configs += 'd_lr = %f\n' % FLAGS.d_lr
  configs += 'g_lr = %f\n' % FLAGS.g_lr
  configs += 'ez_lr = %f\n' % FLAGS.ez_lr
  configs += 'batch_size = %d\n' % FLAGS.batch_size
  configs += 'use_scaling = %s\n' % str(FLAGS.use_scaling)
  configs += 'num_crops = %d\n' % FLAGS.num_crops
  configs += 'app_vector_size = %d\n' % FLAGS.app_vector_size
  configs += 'z_mean = %f\n' % FLAGS.z_mean
  configs += 'z_std = %f\n' % FLAGS.z_std
  configs += 'adam_beta1 = %f\n' % FLAGS.adam_beta1
  configs += 'adam_beta2 = %f\n' % FLAGS.adam_beta2
  configs += '\n# --------------------------------------------------------\n\n'

  configs += ('## Loss weights:\n'
              '## -------------\n')
  configs += 'loss_g_gan = %s\n' % FLAGS.loss_g_gan
  configs += 'loss_direct_recon = %s\n' % FLAGS.loss_direct_recon
  configs += 'loss_cyclic_recon = %s\n' % FLAGS.loss_cyclic_recon
  configs += 'loss_D_direct = %s\n' % FLAGS.loss_D_direct
  configs += 'loss_D_cyclic = %s\n' % FLAGS.loss_D_cyclic
  configs += 'loss_D_rand_z = %s\n' % FLAGS.loss_D_rand_z
  configs += 'loss_D_swap_z = %s\n' % FLAGS.loss_D_swap_z
  configs += 'loss_z_recon = %s\n' % FLAGS.loss_z_recon
  configs += 'loss_z_rand_recon = %s\n' % FLAGS.loss_z_rand_recon
  configs += 'loss_z_kl = %s\n' % FLAGS.loss_z_kl
  configs += 'loss_z_l2 = %s\n' % FLAGS.loss_z_l2
  configs += 'use_vae = %s\n' % str(FLAGS.use_vae)
  configs += 'w_loss_vgg = %f\n' % FLAGS.w_loss_vgg
  configs += 'w_loss_feat = %f\n' % FLAGS.w_loss_feat
  configs += 'w_loss_l1 = %f\n' % FLAGS.w_loss_l1
  configs += 'w_loss_z_recon = %f\n' % FLAGS.w_loss_z_recon
  configs += 'w_loss_gan = %f\n' % FLAGS.w_loss_gan
  configs += 'w_loss_z_gan = %f\n' % FLAGS.w_loss_z_gan
  configs += 'w_loss_z_kl = %f\n' % FLAGS.w_loss_z_kl
  configs += 'w_loss_z_l2 = %f\n' % FLAGS.w_loss_z_l2
  configs += '\n# --------------------------------------------------------\n\n'

  configs += ('## Architecture and training setup:\n'
              '## --------------------------------\n')
  configs += 'model_name = %s\n' % FLAGS.model_name
  configs += 'training_pipeline = %s\n' % FLAGS.training_pipeline
  configs += 'g_nf = %d\n' % FLAGS.g_nf
  configs += 'd_nf = %d\n' % FLAGS.d_nf
  configs += 'num_d_scales = %d\n' % FLAGS.num_d_scales
  configs += 'num_d_layers = %d\n' % FLAGS.num_d_layers
  configs += 'd_activation = %s\n' % FLAGS.d_activation
  configs += 'use_single_disc = %s\n' % str(FLAGS.use_single_disc)
  configs += 'concatenate_skip_layers = %s\n' % str(
                                                FLAGS.concatenate_skip_layers)
  configs += 'pggan_n_blocks= %d\n' % FLAGS.pggan_n_blocks
  configs += 'p2p_n_downsamples = %d\n' % FLAGS.p2p_n_downsamples
  configs += 'p2p_n_resblocks = %d\n' % FLAGS.p2p_n_resblocks
  configs += 'use_concat = %s\n' % str(FLAGS.use_concat)
  configs += 'use_bicycle_gan_ez = %s\n' % str(FLAGS.use_bicycle_gan_ez)
  configs += 'normalize_drit_Ez = %s\n' % str(FLAGS.normalize_drit_Ez)
  configs += 'concat_z_in_all_layers = %s\n' % str(FLAGS.concat_z_in_all_layers)
  configs += 'inject_z = %s\n' % FLAGS.inject_z
  configs += 'use_vgg_loss = %s\n' % str(FLAGS.use_vgg_loss)
  configs += 'z_recon = %s\n' % str(FLAGS.z_recon)
  configs += 'lower_loss_g2_lrn = %s\n' % str(FLAGS.lower_loss_g2_lrn)
  configs += 'use_dropout = %s\n' % str(FLAGS.use_dropout)
  configs += '\n# --------------------------------------------------------\n\n'

  return configs
