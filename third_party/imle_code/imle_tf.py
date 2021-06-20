'''
Implentation of the noise-to-latent mapping approach used in the
"Non-Adversarial Image Synthesis wit Generative Latent Nearest Neighbors" paper,
which can be found at https://arxiv.org/abs/1812.08985.
'''

import collections
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import os.path as osp
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.autograd import Variable
import sys
from sklearn.neighbors import NearestNeighbors
import tensorflow as tf
import time


Hyperparams = collections.namedtuple(
    'Hyperarams',
    'base_lr batch_size num_epochs decay_step decay_rate staleness num_samples_factor')
Hyperparams.__new__.__defaults__ = (None, None, None, None, None, None, None)


# class MLPImplicitModelPytorch(nn.Module):
#   def __init__(self, z_dim, num_layers=3):
#     super(MLPImplicitModel, self).__init__()
#     self.z_dim = z_dim
#     self.num_layers = num_layers
#     # self.mlp1 = nn.Linear(z_dim, z_dim, bias=True)
#     self.mlp2 = nn.Linear(z_dim, 16, bias=True)
#     self.mlp3 = nn.Linear(16, 16, bias=True)
#     self.mlp4 = nn.Linear(16, 16, bias=True)
#     self.mlp5 = nn.Linear(16, z_dim, bias=True)
#     # self.act = nn.ReLU()
#     # self.act = nn.LeakyReLU(0.2)
#     self.act = nn.Tanh()
# 
#   def forward(self, z):
#     # z = self.act(self.mlp1(z))
#     z = self.act(self.mlp2(z))
#     z = self.act(self.mlp3(z))
#     z = self.act(self.mlp4(z))
#     z = self.mlp5(z)
#     return z


# class MLPImplicitModel():
class MLPImplicitModel(tf.keras.Model):
  def __init__(self,
               z_dim,
               num_hidden_layers=3,
               hidden_dim=128,
               act=tf.nn.tanh,  # tf.nn.leakey_relu,
               name='imle'):
    """Initializes the IMLE model.

    Args:
      z_dim: An integer specifying the latent space dimentionality.
      num_hidden_layers: An integer specifying the number of hidden MLP layers.
      hidden_dim: An integer specifying the hidden MLP layers dimentionality,
        this number is used only if 'num_hidden_layers' > 0.
      act: Activation function for hidden MLP layers.
      name: A string specifying a name for the model.
    """
    super(MLPImplicitModel, self).__init__(name=name)
    self.z_dim = z_dim
    self.num_hidden_layers = num_hidden_layers
    self.hidden_dim = hidden_dim
    self._blocks = []
    # Hidden MLP layers with non-linearities.
    for _ in range(num_hidden_layers):
      self._blocks.append(tf.keras.layers.Dense(
          hidden_dim, activation=act, use_bias=True)
      )
    # Final linear MLP
    self._blocks.append(tf.keras.layers.Dense(
        z_dim, activation=None, use_bias=True))

  def __call__(self, z):
    for block_fn in self._blocks:
      z = block_fn(z)
    return z


class IMLE():
  def __init__(self,
               z_dim,
               num_hidden_layers=3,
               hidden_dim=128,
               act=tf.nn.tanh):
    """Builds the IMLE model.

    Args:
      train_dir: A string specifying the train directory to save checkpoints
        and summaries.
      z_dim: An integer specifying the latent space dimentionality.
      num_hidden_layers: An integer specifying the number of hidden MLP layers.
      hidden_dim: An integer specifying the hidden MLP layers dimentionality,
        this number is used only if 'num_hidden_layers' > 0.
      act: Activation function for hidden MLP layers.
    """
    self.z_dim = z_dim
    self.model = MLPImplicitModel(z_dim=z_dim,
                                  num_hidden_layers=num_hidden_layers,
                                  hidden_dim=hidden_dim,
                                  act=act)

  def train(self, train_dir, data_np, hyperparams, data_dir, shuffle_data=True,
            save_checkpoint_secs=300, save_summaries_secs=30,
            noise_perturbation=0.01, log_steps=1):
    loss_fn = tf.nn.l2_loss
    global_step = tf.train.get_or_create_global_step()
    inc_global_step_op = tf.assign(global_step, global_step + 1)
    input_batch_ph = tf.compat.v1.placeholder(
        tf.float32, shape=[None, self.z_dim], name='input')
    gt_batch_ph = tf.compat.v1.placeholder(
        tf.float32, shape=[None, self.z_dim], name='ground_truth')
    output_samples = self.model(input_batch_ph)
    self.input_batch_ph = input_batch_ph 
    self.output_samples = output_samples 
    lr_ph = tf.compat.v1.placeholder(tf.float32, shape=[], name='lr')
    loss = tf.nn.l2_loss(output_samples - gt_batch_ph)
    tf.losses.add_loss(loss)
    tf.summary.scalar('l2_loss', loss, family='losses')
    summary_op = tf.summary.merge_all(name='summary_op')
    optimizer = tf.train.AdamOptimizer(
        learning_rate=lr_ph,
        beta1=0.5,  # TODO: try the default 0.9
        beta2=0.999)
    train_op = tf.group(
        inc_global_step_op,
        optimizer.minimize(loss, var_list=tf.trainable_variables()))

    batch_size = hyperparams.batch_size
    num_batches = data_np.shape[0] // batch_size
    num_samples = num_batches * hyperparams.num_samples_factor

    if shuffle_data:
      data_ordering = np.random.permutation(data_np.shape[0])
      data_np = data_np[data_ordering]
      data_np = data_np[:batch_size*num_batches]  # FIXME: undo truncate some data points, and find a better fix.

    # Reshape to [N, z_dim], where N is the size of the dataset.
    data_np = np.reshape(
        data_np, (data_np.shape[0], np.prod(data_np.shape[1:])))
    # TODO: Maybe try mean normalization, and adding the mean back after inference.
    # data_mean = np.mean(data_np, axis=0)
    # data_np_normalized = data_np - data_mean

    # with tf.train.MonitoredTrainingSession(
    #     checkpoint_dir=train_dir,
    #     save_checkpoint_secs=save_checkpoint_secs,
    #     save_summaries_secs=save_summaries_secs,
    #     log_step_count_steps=log_steps) as sess:
      # while not sess.should_stop():
      # sess.run(train_op)
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())

      # rand_samples = self.sample(len(data_np) // 5)
      rand_samples = self.sample(len(data_np) // 10, sess)
      plt.scatter(data_np[:, 0], data_np[:, 1], c='g')
      plt.scatter(rand_samples[:, 0], rand_samples[:, 1], c='r')
      plt.savefig(osp.join(data_dir, 'before_training.png'))
      # plt.xlim(-5, 10)
      # plt.ylim(-5, 10)
      np.save(osp.join(data_dir, 'rand_samples.npy'), rand_samples)
      plt.close()

      # lm_num_samples = 2
      for epoch in range(hyperparams.num_epochs):

        if epoch % hyperparams.decay_step == 0:
          lr = hyperparams.base_lr * hyperparams.decay_rate ** (
              epoch // hyperparams.decay_step)
          print('lr = %.8f' % lr)
          # optimizer = optim.Adam(self.model.parameters(), lr=lr,
          #                        betas=(0.5, 0.999), weight_decay=1e-5)

        if epoch % hyperparams.staleness == 0:
          z_np = np.empty((num_samples * batch_size, self.z_dim))
          samples_np = np.empty((num_samples * batch_size, self.z_dim))
          for i in range(num_samples):
            z = np.random.normal(size=(batch_size, self.z_dim))
            samples = sess.run(self.output_samples, feed_dict={input_batch_ph: z})
            z_np[i*batch_size:(i+1)*batch_size] = z
            samples_np[i*batch_size:(i+1)*batch_size] = samples

          st = time.time()
          # nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(samples_np)
          # nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree', metric='euclidean').fit(samples_np)
          # nbrs = NearestNeighbors(n_neighbors=1, algorithm='brute', metric='minkowski').fit(samples_np)
          nbrs = NearestNeighbors(n_neighbors=1, algorithm='brute', metric='euclidean').fit(samples_np)
          # nbrs = NearestNeighbors(n_neighbors=1, algorithm='brute', metric='cosine').fit(samples_np)
          fit_time = time.time() - st
          print('kNN fitting took %.2f' % fit_time)
          st = time.time()
          _, nearest_indices = nbrs.kneighbors(data_np)
          query_time = time.time() - st
          print('kNN query took %.2f' % query_time)
          nearest_indices = np.array(nearest_indices)[:,0]
          print(nearest_indices.shape)
          # print(nearest_indices)
          print(len(nearest_indices), len(np.unique(nearest_indices)))

          # print('z_np.shape = %s' % str(z_np.shape))
          z_np = z_np[nearest_indices,]
          # print('z_np.shape = %s' % str(z_np.shape))
          z_np += noise_perturbation * np.random.randn(*z_np.shape)  # FIXME: uncomment this line back!

          del samples_np

        err = 0.
        for i in range(num_batches):
          # self.model.zero_grad()
          cur_z = z_np[i*batch_size:(i+1)*batch_size]
          cur_data = data_np[i*batch_size:(i+1)*batch_size]
          # cur_samples = self.model(cur_z)
          # # print('cur_samples.shape = %s' % str(cur_samples.shape))
          # # print('cur_data.shape = %s' % str(cur_data.shape))
          # loss = loss_fn(cur_samples, cur_data)
          # loss.backward()
          # err += loss.item()
          # optimizer.step()
          _, _, loss_val = sess.run([train_op, summary_op, loss], feed_dict={
              lr_ph: lr, input_batch_ph: cur_z, gt_batch_ph: cur_data})
          err += loss_val

        print("Epoch %d: Error: %f" % (epoch, err / num_batches))
        # rand_samples = self.sample(len(data_np) // 5)
        rand_samples = self.sample(len(data_np) // 10, sess)
        plt.scatter(data_np[:, 0], data_np[:, 1], c='g')
        plt.scatter(rand_samples[:, 0], rand_samples[:, 1], c='r')
        plt.savefig(osp.join(data_dir, 'epoch_%d.png' % epoch))
        # plt.xlim(-5, 10)
        # plt.ylim(-5, 10)
        np.save(osp.join(data_dir, 'rand_samples.npy'), rand_samples)
        plt.close()
    
      final_samples = self.sample(64, sess)
      np.save(osp.join(data_dir, 'rand_samples.npy'), final_samples)
      saver = tf.compat.v1.train.Saver()
      saver.save(sess, 'mapper')


  def sample(self, num_samples, sess):
    z_np = np.random.normal(size=(num_samples, self.z_dim))
    samples_np = sess.run(self.output_samples, feed_dict={self.input_batch_ph: z_np})
    return samples_np


def main(*args):
  # ---------------------------------------------------------------------------
  # dataset_name = 'edges2handbags'
  # data_dir = '/vulcan/scratch/mmeshry/appearance_pretraining/train/edges2handbags/custom/edges2handbags-custom_staged_v2-pretrain_handbags-finetune/exported_styles'
  # ---------------------------------------------------------------------------
  # dataset_name = 'edges2shoes'
  data_dir = '/vulcan/scratch/mmeshry/appearance_pretraining/train/edges2shoes/custom/edges2shoes-custom_staged_v2-pretrain_handbags-r2-finetune/exported_styles'
  # data_dir = '/vulcan/scratch/mmeshry/appearance_pretraining/train/edges2shoes/custom/edges2shoes-custom_staged_v3-pretrain_handbags-finetune/exported_styles'
  # data_dir = '/vulcan/scratch/mmeshry/appearance_pretraining/train/edges2shoes/custom/edges2shoes-custom_staged_cross_cycle_z-pretrain_handbags-same_opt_setup-finetune/exported_styles'
  # data_dir = '/vulcan/scratch/mmeshry/appearance_pretraining/train/edges2shoes/custom/edges2shoes-custom_staged_cross_cycle-pretrain_handbags-same_opt_setup-finetune/exported_styles'
  # ---------------------------------------------------------------------------
  # dataset_name = 'facades'
  # data_dir = '/vulcan/scratch/mmeshry/appearance_pretraining/train/facades/custom/facades-custom_staged_v2-pretrain_handbags-nf16-finetune/exported_styles'
  # ---------------------------------------------------------------------------
  # dataset_name = 'maps'
  # data_dir = '/vulcan/scratch/mmeshry/appearance_pretraining/train/maps/custom/maps-custom_staged_v2-finetune/exported_styles'
  # ---------------------------------------------------------------------------
  # dataset_name = 'night2day'
  # data_dir = '/vulcan/scratch/mmeshry/appearance_pretraining/train/night2day/custom/night2day-custom_staged_v2-pretrain_space_needle-nf16-finetune/exported_styles'
  # ---------------------------------------------------------------------------
  # dataset_name = 'space_needle_timelapse'
  # data_dir = '/vulcan/scratch/mmeshry/appearance_pretraining/train/space_needle_timelapse/custom/space_needle_timelapse-custom_staged_v2-pretrain_space_needle-finetune/exported_styles'
  # ---------------------------------------------------------------------------

  # train_data = np.load(osp.join(data_dir, 'zs_train.npy'))
  # if train_data.shape[0] > 50000:
  #   print('*** Warning: shuffling and truncating dataset to 50K records only!')
  #   idxs = np.random.permutation(train_data.shape[0])
  #   train_data = train_data[idxs[:50000], :]
  # z_dim = 8
  # ---------------------------------------------------------------------------

  dataset_name = 'toy_dataset'
  data_dir = '/vulcan/scratch/mmeshry/appearance_pretraining/imle_tests_tf-temp'
  data_size = 2000
  # mu = (.5, 5)
  # std = (2, 0.2)
  # mu1 = (5, 4)  # (-5, -0)
  # mu2 = (-1, -2)  # (5, 0)
  mu1 = (-5, -0)
  mu2 = (5, 0)
  std1 = (1, 1)
  std2 = (1, 1)
  cluster1 = np.random.normal(mu1, std1, (data_size // 2, 2))
  cluster2 = np.random.normal(mu2, std2, (data_size // 2, 2))
  cluster1 = cluster1 / np.sqrt(np.sum(np.square(cluster1), axis=1))[..., np.newaxis]
  cluster2 = cluster2 / np.sqrt(np.sum(np.square(cluster2), axis=1))[..., np.newaxis]
  train_data = np.vstack((cluster1, cluster2))
  z_dim = 2
  # ---------------------------------------------------------------------------

  imle = IMLE(z_dim,
              num_hidden_layers=3,
              hidden_dim=128,
              act=tf.nn.tanh)
  vars_all = tf.trainable_variables()

  # Hyperparameters:

  # base_lr: Base learning rate
  # batch_size: Batch size
  # num_epochs: Number of epochs
  # decay_step: Number of epochs before learning rate decay
  # decay_rate: Rate of learning rate decay
  # staleness: Number of times to re-use nearest samples
  # num_samples_factor: Ratio of the number of generated samples to the number of real data examples

  if not osp.exists(data_dir):
    os.makedirs(data_dir)
  imle.train(data_dir, train_data, Hyperparams(
                base_lr=1e-2,  # 1e-2, 1e-3
                batch_size=64,
                num_epochs=1000,  # 1000, 400
                decay_step=50,  # 25, 50, 100
                # decay_rate=1.0, staleness=5, num_samples_factor=10), data_dir)
                decay_rate=0.7,  # 1., 0.5, 0.7
                staleness=5,
                num_samples_factor=10,  # 10, 1
             ),
             data_dir)
                # decay_rate=1.0, staleness=1, num_samples_factor=1), data_dir)

  # torch.save(imle.model.state_dict(),
  #            osp.join(data_dir, '%s_mapper_weights.pth' % dataset_name))
  # TODO: FIXME load trained weights, and run inference in a session! Maybe set a global session and manually save checkpoints!
  # samples = imle.sample(64)
  # np.save(osp.join(data_dir, 'rand_samples.npy'), samples)


if __name__ == '__main__':
  main(*sys.argv[1:])
