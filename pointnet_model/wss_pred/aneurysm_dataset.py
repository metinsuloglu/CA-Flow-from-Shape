"""
Dataset of CA morphologies
"""

import os
import os.path
import json
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import provider
import numpy as np
from scipy.stats import ortho_group
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf

def pc_resample(pcs, tawss, npoints=1024, add_type='linear'):
  new_pcs, new_tawss = [], []
  for i in range(len(pcs)):
    if pcs[i].shape[0] == npoints:
      new_pcs.append(pcs[i])
      new_tawss.append(tawss[i])
    elif pcs[i].shape[0] > npoints:
      choice = np.random.choice(pcs[i].shape[0], npoints, replace=False)
      new_pcs.append(pcs[i][choice])
      new_tawss.append(tawss[i][choice])
    else:
      if add_type == 'repeat':
        factor = np.ceil(npoints/float(pcs[i].shape[0]));
        choice = np.tile(range(pcs[i].shape[0]), int(factor))
        new_pcs.append(pcs[i][choice[:npoints], :])
        new_tawss.append(tawss[i][choice[:npoints]])
      elif add_type == 'linear':
        kd = KDTree(pcs[i])
        selected_points = np.random.choice(list(range(pcs[i].shape[0])), size=npoints - pcs[i].shape[0], replace=True)
        nns = kd.query(pcs[i][selected_points,:], k=2)[1][:,1]
        rand_skip = np.random.rand(pcs[i][selected_points,:].shape[0], 1)
        new_pcs.append(np.vstack((pcs[i], pcs[i][selected_points,:] + rand_skip * np.subtract(pcs[i][nns,:], pcs[i][selected_points,:]))))
        new_tawss.append(np.hstack((tawss[i], tawss[i][selected_points] + rand_skip.squeeze() * np.subtract(tawss[i][nns], tawss[i][selected_points]))))
      else: print('Unknown upsampling type!')
  return new_pcs, new_tawss


def pc_normalise(cases, scale=None):
  """
  Normalises the cases such that centroids are at the origin and 
  the max distance of a point on the surface to the origin is 'scale'.
  If scale is None then no scaling is performed.
  """
  assert(cases.shape[2] == 3)
  new_points = cases - cases.mean(axis=1)[:, np.newaxis, :]
  if scale is not None:
    max_norm = np.amax(np.linalg.norm(new_points, axis=2), axis=1)
    new_points = new_points / max_norm[:, np.newaxis, np.newaxis] * scale
  return new_points


def interpolate3d(points, indices, func_vals, **kwargs):
  """
  Interpolate values on 3D point clouds.
  """
  rbfi = Rbf(points[indices,0], points[indices,1], points[indices,2],
             func_vals, **kwargs)
  oth_ind = list(set(range(len(points))) - set(indices))
  di = rbfi(points[oth_ind,0], points[oth_ind,1], points[oth_ind,2])
  new_vals = np.zeros(len(points))
  new_vals[indices] = func_vals
  new_vals[oth_ind] = di
  return new_vals


class AneurysmDataset():
  def __init__(self, root, fnames, npoints = 1024, max_norm=None, wss_min_max=None, rand_aug=None):
    """
    root: data directory
    fnames: the file names of the data samples to be used, must be provided
    npoints: number of points in each point cloud
    max_norm: scaling factor. set to None for training set, and provide the training set max norm for val and test sets
    wss_min_max: min-max tawss values from the training set, used for normalisation
    rand_aug: number of augmented copies of the initial dataset to add
    """

    # Dummy values to check for bugs
    #self.point_sets = np.random.uniform(size=(100,512,3))
    #self.tawss_vals = self.point_sets[:,:,3].squeeze()
    #self.ids = ['dummy_%d'.format(x) for x in range(self.point_sets.shape[0])]
    #return

    assert(fnames is not None)

    # Load data
    cases, self.ids = [], []
    for f in fnames:
      assert f.endswith('.txt')
      cases.append(np.loadtxt(os.path.join(root, f), delimiter=None))
      self.ids.append(f)

    # Only keep unique points
    for i in range(len(cases)):
      cases[i] = np.unique(cases[i], axis=0)

    # Shuffle points
    for i in range(len(cases)): np.random.shuffle(cases[i])
    
    self.tawss_vals = [x[:,3] for x in cases]
    self.point_sets = [x[:,:3] for x in cases]

    # RBF interpolation
    for i in range(len(self.tawss_vals)):
      assigned_idxs = np.where(self.tawss_vals[i]!=-1)[0]
      self.tawss_vals[i] = interpolate3d(self.point_sets[i], assigned_idxs, self.tawss_vals[i][assigned_idxs], function='linear')

    # Standardise the number of points
    self.point_sets, self.tawss_vals = pc_resample(self.point_sets, self.tawss_vals, npoints)
    self.point_sets, self.tawss_vals = np.array(self.point_sets), np.vstack(self.tawss_vals)

    # Random augmentations
    #if rand_aug is not None:
    #  new_point_sets = []
    #  for r in range(rand_aug-1):
    #    aug_data = provider.rotate_perturbation_point_cloud(self.point_sets[:,:,:3])
    #    aug_data = provider.jitter_point_cloud(aug_data)
    #    aug_data = provider.shift_point_cloud(aug_data)
    #    new_point_sets.append(np.dstack((aug_data, self.point_sets[:,:,3:])))
    #  self.point_sets = np.vstack([self.point_sets] + new_point_sets)
    #  self.tawss_vals = np.vstack([self.tawss_vals] + [self.tawss_vals]*rand_aug)
    #  self.ids = self.ids * (rand_aug + 1)

    # Normalise to centre
    self.point_sets = pc_normalise(self.point_sets, scale=None)
    # Scale cases proportionally
    if max_norm is None:
      self.max_norm = np.amax(np.linalg.norm(self.point_sets, axis=2))
      self.point_sets = self.point_sets / self.max_norm
    else:
      self.point_sets = self.point_sets / max_norm
    # Calculate current norms
    norms = np.repeat(np.amax(np.linalg.norm(self.point_sets, axis=2), axis=1).reshape(-1,1), npoints, axis=1)
    # Scale all cases so max_norm is 1.0
    self.point_sets = pc_normalise(self.point_sets, scale=1.0)
    # Append previous norms to new scaled cases
    self.point_sets = np.dstack((self.point_sets, norms))
    
    # Unskew TAWSS values
    self.tawss_vals = np.cbrt(self.tawss_vals)
    if wss_min_max is None: self.wss_min_max = (self.tawss_vals.min(), self.tawss_vals.max())
    else: self.wss_min_max = (wss_min_max[0], wss_min_max[1])
    # Min-max normalisation:
    self.tawss_vals = (self.tawss_vals - self.wss_min_max[0]) / (self.wss_min_max[1] - self.wss_min_max[0])
      

  def __getitem__(self, index):
    return self.point_sets[index,:,:], self.tawss_vals[index,:], self.ids[index]
      

  def __len__(self):
    return self.point_sets.shape[0]


  def untransform(self, preds):
    cbrt_preds = preds * (self.wss_min_max[1] - self.wss_min_max[0]) + self.wss_min_max[0]
    return np.power(cbrt_preds, 3)

