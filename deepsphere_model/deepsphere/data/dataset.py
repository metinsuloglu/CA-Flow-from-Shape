"""
Dataset processing for CA shapes
"""

import os

import numpy as np
from scipy.stats import special_ortho_group
from torch.utils.data import Dataset
from data.process import map_to_sphere, normalise

class WSSSphereLoader(Dataset):
    """
    Loader for a dataset of CA morphologies.
    """

    def __init__(self, path, fnames=None, partition='train', transform_data=None, transform_labels=None, transform_mask=None, npix=None, max_norm=None, wss_mean_std=None):
        """
        Initialization.

        Args:
            path (str): Path to the data.
            fnames (list): Files names to load.
            partition: train val or test (name)
            transform_data (:obj:`transform.Compose`): List of torchvision transforms for the data.
            transform_labels (:obj:`transform.Compose`): List of torchvision transforms for the labels.
            transform_mask (:obj:`transform.Compose`): List of torchvision transforms for the binary mask / weights.
            npix: Number of vertices on the icosphere
            max_norm: Used to normalise the CA shapes
            wss_mean_std: Used for mean normalisation of WSS values
        """
        self.path = path
        self.partition = partition
        data = []
        self.files = []
        for f in fnames:
            assert f.endswith('.txt')
            data.append(np.loadtxt(os.path.join(path, f), delimiter=None))
            self.files.append(f)
           
        # Split coords and outputs
        xyz = [x[:,:3] for x in data]
        wss = [x[:,3] for x in data]
        
        wss = [np.cbrt(x) for x in wss] # cbrt transform
        trans_xyz = normalise(xyz, scale=None) # translate shapes to the origin
        
        # Random rotations
        #if partition =='train':
        #   for i in range(100):
        #       trans_xyz.append(np.matmul(trans_xyz[i%len(indices)], special_ortho_group.rvs(3).T))
        #       wss.append(wss[i%len(indices)])
        
        # Scale cases proportionally
        if max_norm is None:
            self.max_norm = max([max(np.linalg.norm(c, axis=1)) for c in trans_xyz])
        else:
            self.max_norm = max_norm
        self.norm_xyz = [c / self.max_norm for c in trans_xyz]
        
        # shapes = (n_samples, n_points)
        self.dists, self.wss_map, self.sphere, self.binary_map, self.corrs = map_to_sphere(self.norm_xyz, wss, theta=0.08,
                                                                                           interp_type='linear', npix=npix)
        
        if wss_mean_std is None: self.wss_mean_std = (self.wss_map.mean(), self.wss_map.std())
        else: self.wss_mean_std = wss_mean_std
        self.wss_map = (self.wss_map - self.wss_mean_std[0]) / self.wss_mean_std[1]
        
        self.binary_map = self.binary_map / self.binary_map.sum(axis=1, keepdims=True) # weight matrix
        
        assert(len(self.dists) == len(self.wss_map))
        
        self.transform_data = transform_data
        self.transform_labels = transform_labels
        self.transform_mask = transform_mask

    @property
    def indices(self):
        """
        Get files.

        Returns:
            list: List of strings, which represent the files contained in the dataset.
        """
        return self.files

    def __len__(self):
        """
        Get length of dataset.

        Returns:
            int: Number of files contained in the dataset.
        """
        return len(self.dists)

    def __getitem__(self, idx):
        """
        Get an item from the dataset.

        Args:
            idx (int): The index of the desired datapoint.

        Returns:
            obj, obj, obj: The data, labels and weights corresponding to the desired index. The type depends on the applied transforms.
        """
        data = self.dists[None, idx, :].astype(np.float32) # (1, n_points)
        vals = self.wss_map[None, idx, :].astype(np.float32) # (1, n_points)
        m = self.binary_map[None, idx, :].astype(np.float32) # (1, n_points)
        if self.transform_data:
            data = self.transform_data(data)
        if self.transform_labels:
            vals = self.transform_labels(vals)
        if self.transform_mask:
            m = self.transform_mask(m)
        return data, vals, m
        
    def untransform(self, vals):
        cbrt_preds = vals * self.wss_mean_std[1] + self.wss_mean_std[0]
        return np.power(cbrt_preds, 3)
