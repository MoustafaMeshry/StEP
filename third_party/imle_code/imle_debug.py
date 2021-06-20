'''
Code for Implicit Maximum Likelihood Estimation

This code implements the method described in the Implicit Maximum Likelihood
Estimation paper, which can be found at https://arxiv.org/abs/1809.09087

Copyright (C) 2018    Ke Li


This file is part of the Implicit Maximum Likelihood Estimation reference
implementation.

The Implicit Maximum Likelihood Estimation reference implementation is free
software: you can redistribute it and/or modify it under the terms of the GNU
Affero General Public License as published by the Free Software Foundation,
either version 3 of the License, or (at your option) any later version.

The Implicit Maximum Likelihood Estimation reference implementation is
distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with the Dynamic Continuous Indexing reference implementation.  If
not, see <http://www.gnu.org/licenses/>.
'''

import numpy as np
import os.path as osp
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import sys
# sys.path.append('./third_party/imle_code/dci_code')
sys.path.append('./dci_code')
# from dci import DCI
from sklearn.neighbors import NearestNeighbors
import collections
import time
import matplotlib.pyplot as plt


Hyperparams = collections.namedtuple(
    'Hyperarams',
    'base_lr batch_size num_epochs decay_step decay_rate staleness num_samples_factor')
Hyperparams.__new__.__defaults__ = (None, None, None, None, None, None, None)


class MLPImplicitModel(nn.Module):
  def __init__(self, z_dim, num_layers=3):
    super(MLPImplicitModel, self).__init__()
    self.z_dim = z_dim
    self.num_layers = num_layers
    # self.mlp1 = nn.Linear(z_dim, z_dim, bias=True)
    self.mlp2 = nn.Linear(z_dim, 16, bias=True)
    self.mlp3 = nn.Linear(16, 16, bias=True)
    self.mlp4 = nn.Linear(16, 16, bias=True)
    self.mlp5 = nn.Linear(16, z_dim, bias=True)
    # self.act = nn.ReLU()
    # self.act = nn.LeakyReLU(0.2)
    self.act = nn.Tanh()

  def forward(self, z):
    # z = self.act(self.mlp1(z))
    z = self.act(self.mlp2(z))
    z = self.act(self.mlp3(z))
    z = self.act(self.mlp4(z))
    z = self.mlp5(z)
    return z


class IMLE():
  def __init__(self, z_dim):
    self.z_dim = z_dim
    self.model = MLPImplicitModel(z_dim).cuda()
    self.dci_db = None

  def train(self, data_np, hyperparams, data_dir, shuffle_data=True):
    loss_fn = nn.MSELoss().cuda()
    self.model.train()

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
    # data_mean = np.mean(data_np, axis=0)
    # data_np_normalized = data_np - data_mean

    # # Initialize the nearest neighbor instance.
    # if self.dci_db is None:
    #   self.dci_db = DCI(np.prod(data_np.shape[1:]), num_comp_indices=2,
    #                     num_simp_indices=7)

    # lm_num_samples = 2
    for epoch in range(hyperparams.num_epochs):

      if epoch % hyperparams.decay_step == 0:
        lr = hyperparams.base_lr * hyperparams.decay_rate ** (
            epoch // hyperparams.decay_step)
        print('lr = %.8f' % lr)
        optimizer = optim.Adam(self.model.parameters(), lr=lr,
                               betas=(0.5, 0.999), weight_decay=1e-5)

      if epoch % hyperparams.staleness == 0:
        z_np = np.empty((num_samples * batch_size, self.z_dim))
        samples_np = np.empty((num_samples * batch_size, self.z_dim))
        for i in range(num_samples):
          z = torch.randn(batch_size, self.z_dim).cuda()
          samples = self.model(z)
          z_np[i*batch_size:(i+1)*batch_size] = z.cpu().data.numpy()
          samples_np[i*batch_size:(i+1)*batch_size] = samples.cpu().data.numpy()
        # z_np = np.empty((num_samples * batch_size, self.z_dim))
        # samples_np = np.empty((num_samples * batch_size, self.z_dim))
        # z = torch.randn(lm_num_samples, self.z_dim).cuda()
        # samples = self.model(z)
        # z_ar = z.cpu().data.numpy()
        # samples_ar = samples.cpu().data.numpy()
        # num_repeat = len(data_np) // len(z_ar) + 1
        # z_ar = np.tile(z_ar, (num_repeat, 1))
        # samples_ar = np.tile(samples_ar, (num_repeat, 1))
        # for i in range(num_batches):
        #   z_np[i*batch_size:(i+1)*batch_size] = z_ar[i*batch_size:(i+1)*batch_size]
        #   samples_np[i*batch_size:(i+1)*batch_size] = samples_ar[i*batch_size:(i+1)*batch_size] 

        # self.dci_db.reset()
        # self.dci_db.add(samples_np, num_levels=2, field_of_view=10,
        #                 prop_to_retrieve=0.002)
        # nearest_indices, _ = self.dci_db.query(
        #     data_np, num_neighbours=1, field_of_view=20,
        #     prop_to_retrieve=0.02)
        # nearest_indices = np.array(nearest_indices)[:, 0]

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
        z_np += 0.01 * np.random.randn(*z_np.shape)  # FIXME: uncomment this line back!

        del samples_np

      err = 0.
      for i in range(num_batches):
        self.model.zero_grad()
        cur_z = torch.from_numpy(
            z_np[i*batch_size:(i+1)*batch_size]).float().cuda()
        cur_data = torch.from_numpy(
            data_np[i*batch_size:(i+1)*batch_size]).float().cuda()
        cur_samples = self.model(cur_z)
        # print('cur_samples.shape = %s' % str(cur_samples.shape))
        # print('cur_data.shape = %s' % str(cur_data.shape))
        loss = loss_fn(cur_samples, cur_data)
        loss.backward()
        err += loss.item()
        optimizer.step()

      print("Epoch %d: Error: %f" % (epoch, err / num_batches))
      # rand_samples = self.sample(len(data_np) // 5)
      rand_samples = self.sample(len(data_np) // 10)
      plt.scatter(data_np[:, 0], data_np[:, 1], c='g')
      plt.scatter(rand_samples[:, 0], rand_samples[:, 1], c='r')
      plt.savefig(osp.join(data_dir, 'epoch_%d.png' % epoch))
      # plt.xlim(-5, 10)
      # plt.ylim(-5, 10)
      np.save(osp.join(data_dir, 'rand_samples.npy'), rand_samples)
      plt.close()


  def sample(self, num_samples):
    z_np = np.empty((num_samples, self.z_dim))
    samples_np = np.empty((num_samples, self.z_dim))
    z = torch.randn(num_samples, self.z_dim).cuda()
    samples = self.model(z)
    z_np[:] = z.cpu().data.numpy()
    samples_np[:] = samples.cpu().data.numpy()
    return samples_np


def main(*args):

  device_id = 0
  torch.cuda.set_device(device_id)
  dataset_name = 'edges2handbags'
  data_dir = '/vulcan/scratch/mmeshry/appearance_pretraining/train/edges2handbags/custom/edges2handbags-custom_staged_v2-pretrain_handbags-finetune/exported_styles'
  # dataset_name = 'edges2shoes'
  # data_dir = '/vulcan/scratch/mmeshry/appearance_pretraining/train/edges2shoes/custom/edges2shoes-custom_staged_v2-pretrain_handbags-r2-finetune/exported_styles'
  # dataset_name = 'facades'
  # data_dir = '/vulcan/scratch/mmeshry/appearance_pretraining/train/facades/custom/facades-custom_staged_v2-pretrain_handbags-nf16-finetune/exported_styles'
  # dataset_name = 'maps'
  # data_dir = '/vulcan/scratch/mmeshry/appearance_pretraining/train/maps/custom/maps-custom_staged_v2-finetune/exported_styles'
  # dataset_name = 'night2day'
  # data_dir = '/vulcan/scratch/mmeshry/appearance_pretraining/train/night2day/custom/night2day-custom_staged_v2-pretrain_space_needle-nf16-finetune/exported_styles'
  # dataset_name = 'space_needle_timelapse'
  # data_dir = '/vulcan/scratch/mmeshry/appearance_pretraining/train/space_needle_timelapse/custom/space_needle_timelapse-custom_staged_v2-pretrain_space_needle-finetune/exported_styles'

  # # train_data is of shape N x z_dim, where N is the number of dataset examples.
  # train_data = np.load(osp.join(data_dir, 'zs_train.npy'))
  # if train_data.shape[0] > 50000:
  #   print('*** Warning: shuffling and truncating dataset to 50K records only!')
  #   idxs = np.random.permutation(train_data.shape[0])
  #   train_data = train_data[idxs[:50000], :]
  # z_dim = 8

  dataset_name = 'toy_dataset'
  data_dir = '/vulcan/scratch/mmeshry/appearance_pretraining/imle_tests'
  data_size = 2000
  # mu = (.5, 5)
  # std = (2, 0.2)
  mu1 = (5, 4)  # (-5, -0)
  mu2 = (-1, -2)  # (5, 0)
  std1 = (1, 1)
  std2 = (1, 1)
  cluster1 = np.random.normal(mu1, std1, (data_size // 2, 2))
  cluster2 = np.random.normal(mu2, std2, (data_size // 2, 2))
  # cluster1 = cluster1 / np.sqrt(np.sum(np.square(cluster1), axis=1))[..., np.newaxis]
  # cluster2 = cluster2 / np.sqrt(np.sum(np.square(cluster2), axis=1))[..., np.newaxis]
  train_data = np.vstack((cluster1, cluster2))
  z_dim = 2

  imle = IMLE(z_dim)

  # Hyperparameters:

  # base_lr: Base learning rate
  # batch_size: Batch size
  # num_epochs: Number of epochs
  # decay_step: Number of epochs before learning rate decay
  # decay_rate: Rate of learning rate decay
  # staleness: Number of times to re-use nearest samples
  # num_samples_factor: Ratio of the number of generated samples to the number of real data examples

  # if not osp.exists(data_dir):
  #   os.make_dirs(data_dir)
  imle.train(train_data, Hyperparams(
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

  torch.save(imle.model.state_dict(),
             osp.join(data_dir, '%s_mapper_weights.pth' % dataset_name))
  samples = imle.sample(64)
  np.save(osp.join(data_dir, 'rand_samples.npy'), samples)


if __name__ == '__main__':
  main(*sys.argv[1:])
