"""
Script for testing a DeepSphere U-Net for predicting TAWSS over a CA point cloud.
"""

import os
import sys
sys.path.append('./deepsphere')

from collections import OrderedDict
import numpy as np
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import transforms

from data.dataset import WSSSphereLoader
from data.transforms.transforms import Permute, ToTensor
from models.spherical_unet.unet_model import SphericalUNet
from utils.initialization import init_device, get_device
from utils.parser import create_parser, parse_config
from utils.metrics import *
from utils.funcs import reconstruct

from utils.vis import plot_cases
from matplotlib import cm
import matplotlib.colors as mcolors

def log(text, parser_args):
    with open(os.path.join(parser_args.model_save_path, 'log.txt'), 'a') as f:
        f.write(text + '\n')
        print(text)
        f.close()

def load_checkpoint(model, load_dict, exclude='none'):
    from torch.nn.parameter import Parameter

    own_state = model.state_dict()
    for name, param in load_dict['state_dict'].items():
        if name not in own_state:
            continue
        if exclude in name:
            continue
        if isinstance(param, Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        own_state[name].copy_(param)
        
    return load_dict['wss_mean_std']

def get_test_dataloader(parser_args, wss_mean_std):
    """
    Creates the test dataset and the corresponding dataloader

    Args:
        parser_args (dict): parsed arguments

    Returns:
        test dataloader
    """

    path_to_data = parser_args.path_to_data
    
    precomp_norm = 0.6843864666470979
    transform_data = transforms.Compose([ToTensor(), Permute()])
    transform_labels = transforms.Compose([ToTensor(), Permute()])
    transform_mask = transforms.Compose([ToTensor(), Permute()])
    test_set = WSSSphereLoader(path=path_to_data, fnames=['case_{}.txt'.format(str(x).zfill(2)) for x in list(range(34,39))],
                               npix=parser_args.n_pixels, transform_data=transform_data,
                               transform_labels=transform_labels, transform_mask=transform_mask,
                               max_norm=precomp_norm, partition='test', wss_mean_std=wss_mean_std)
    dataloader_test = DataLoader(test_set, batch_size=parser_args.batch_size, shuffle=False, num_workers=4, drop_last=False)
    
    return dataloader_test
    
def test(args, model, loader, device):
    model.eval()
    d, y, y_hat, m = [], [], [], []
    with torch.no_grad():
       for batch, (data, labels, mask) in enumerate(loader, 1):
           data, labels, mask = data.to(device), labels.to(device), mask.to(device)
           output = model(data)
           d.append(data)
           y.append(labels)
           y_hat.append(output)
           m.append(mask)
       d, y, y_hat, m = torch.cat(d), torch.cat(y), torch.cat(y_hat), torch.cat(m)
    return d, y, y_hat, m
    

def main(parser_args):
    """
    Main function

    Args:
        parser_args (dict): parsed arguments
    """
    
    unet = SphericalUNet(parser_args.pooling_class, parser_args.n_pixels,
                         parser_args.depth, parser_args.laplacian_type,
                         parser_args.kernel_size)
    
    wss_mean_std = load_checkpoint(unet, torch.load('./pretrained_model_38_cases.pt', map_location=get_device(parser_args.device)))
    unet, device = init_device(parser_args.device, unet)
    dataloader_test = get_test_dataloader(parser_args, wss_mean_std)
    
    d, y, y_hat, m = test(parser_args, unet, dataloader_test, device)
    log('[Test set] - L1: {:.8f}   L2: {:.8f}   Avg. Pearson: {:.8f}'.format(mae(y_hat, y, mask=m, device=device),
                                                                             mse(y_hat, y, mask=m, device=device),
                                                                             avg_pearson(y_hat, y, mask=m, device=device)), parser_args)
    try:
      d, m = d.numpy().squeeze(), m.numpy().squeeze()
      y, y_hat = y.numpy().squeeze(), y_hat.numpy().squeeze()
    except TypeError:
      d, m = d.cpu().numpy().squeeze(), m.cpu().numpy().squeeze()
      y, y_hat = y.cpu().numpy().squeeze(), y_hat.cpu().numpy().squeeze()
    results = {'dists': d, 'corrs': dataloader_test.dataset.corrs,
               'y': dataloader_test.dataset.untransform(y), 'y_hat': dataloader_test.dataset.untransform(y_hat),
               'weights': m
              }
    np.save(os.path.join(parser_args.model_save_path, 'test_results.npy'), results)  # load back with np.load(..., allow_pickle=True).item()
    log('Saving results to file: {}'.format(os.path.join(parser_args.model_save_path, 'test_results.npy')), parser_args)
    
    if False:  # set to True for visualisation of reconstructed CA point clouds and transformed TAWSS values
        for ind in range(len(dataloader_test.dataset)):
            reconstructed = reconstruct(dataloader_test.dataset.sphere.coords[m[ind]!=0], d[ind][m[ind]!=0])
            norm1 = mcolors.Normalize(vmin=min(y[ind][m[ind]!=0]), vmax=max(y[ind][m[ind]!=0]))
            norm2 = mcolors.Normalize(vmin=min(y_hat[ind][m[ind]!=0]), vmax=max(y_hat[ind][m[ind]!=0]))
            plot_cases([reconstructed, reconstructed+1.2],
                        colours=[np.apply_along_axis(cm.rainbow,
                                 0, norm1(y[ind][m[ind]!=0]))[:,:3], np.apply_along_axis(cm.rainbow,
                                 0, norm2(y_hat[ind][m[ind]!=0]))[:,:3]], backend='open3d')

if __name__ == "__main__":
    PARSER_ARGS = parse_config(create_parser())
    main(PARSER_ARGS)
