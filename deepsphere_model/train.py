"""
Script for training a DeepSphere U-Net to predict TAWSS over a CA point cloud.
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
from utils.initialization import init_device
from utils.parser import create_parser, parse_config
from utils.metrics import *

def log(text, parser_args):
    with open(os.path.join(parser_args.model_save_path, 'log.txt'), 'a') as f:
        f.write(text + '\n')
        print(text)
        f.close()

def l1(output, target):
    return torch.mean(torch.abs(output - target))

def weighted_l1(output, target, weight):
    return torch.mean(torch.mul(weight, torch.abs(output - target)))
    
def weighted_l2(output, target, weight):
    return torch.mean(torch.mul(weight, torch.pow((output - target), 2)))

def weighted_huber(output, target, weight, delta=1.345):
    return torch.mean(torch.where(torch.abs(output - target) <= delta , torch.mul(weight, torch.mul(torch.pow((output-target), 2), 0.5)), torch.mul(weight, torch.mul(torch.abs(output - target), delta) - 0.5 * (delta ** 2))))
    
def save_checkpoint(model, epoch, optimizer, wss_mean_std, path):
    state_dict_no_sparse = [it for it in model.state_dict().items() if it[1].type() != "torch.sparse.FloatTensor"]
    state_dict_no_sparse = OrderedDict(state_dict_no_sparse)
    torch.save({'epoch': epoch, 'state_dict': state_dict_no_sparse, 'optimizer': optimizer.state_dict(), 'wss_mean_std': wss_mean_std
    }, path)

def get_dataloaders(parser_args):
    """
    Creates the datasets and the corresponding dataloaders

    Args:
        parser_args (dict): parsed arguments

    Returns:
        train, validation dataloaders and z-score normalisation parameters
    """

    path_to_data = parser_args.path_to_data
    
    precomp_norm = 0.6843864666470979 # used to normalise CA point clouds
    transform_data = transforms.Compose([ToTensor(), Permute()])
    transform_labels = transforms.Compose([ToTensor(), Permute()])
    transform_mask = transforms.Compose([ToTensor(), Permute()])
    train_set = WSSSphereLoader(path=path_to_data, fnames=['case_{}.txt'.format(str(x).zfill(2)) for x in list(range(1,34))],
                                npix=parser_args.n_pixels, transform_data=transform_data,
                                transform_labels=transform_labels, transform_mask=transform_mask,
                                max_norm=precomp_norm, partition='train')
    val_set = WSSSphereLoader(path=path_to_data, fnames=['case_{}.txt'.format(str(x).zfill(2)) for x in list(range(34,39))],
                              npix=parser_args.n_pixels, transform_data=transform_data,
                              transform_labels=transform_labels, transform_mask=transform_mask,
                              max_norm=precomp_norm, partition='val', wss_mean_std=train_set.wss_mean_std)
    dataloader_train = DataLoader(train_set, batch_size=parser_args.batch_size, shuffle=True, num_workers=4, drop_last=False)
    dataloader_val = DataLoader(val_set, batch_size=parser_args.batch_size, shuffle=False, num_workers=4, drop_last=False)
    
    return dataloader_train, dataloader_val, train_set.wss_mean_std

def train(args, model, loader, epoch, optimizer, device):
    model.train()
    train_losses = []
    for batch, (data, labels, mask) in enumerate(loader, 1):
        data, labels, mask = data.to(device), labels.to(device), mask.to(device)
        output = model(data)

        loss = weighted_l1(output, labels, mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch % 2 == 0:
            log('[Train Epoch {} | {}/{} ({:.0f}%)] Loss: {:.8f}'.format(epoch, batch * len(data), len(loader.dataset), 100. * batch / len(loader), loss.item()), args)
        train_losses.append(loss.item())
    return train_losses
    
def validate(args, model, loader, epoch, device, name='Val'):
    model.eval()
    y, y_hat, m = [], [], []
    with torch.no_grad():
        for batch, (data, labels, mask) in enumerate(loader, 1):
            data, labels, mask = data.to(device), labels.to(device), mask.to(device)
            output = model(data)
            y.append(labels)
            y_hat.append(output)
            m.append(mask)
        y, y_hat, m = torch.cat(y), torch.cat(y_hat), torch.cat(m)
        scores = {'l1':mae(y_hat, y, mask=m, device=device),
                  'l2':mse(y_hat, y, mask=m, device=device),
                  'avg_pearson':avg_pearson(y_hat, y, mask=m, device=device)}
        log('[{} Eval Epoch {}] - L1: {:.8f}   L2: {:.8f}   Avg. Pearson: {:.8f}'.format(name, epoch, scores['l1'], scores['l2'], scores['avg_pearson']), args)
    return scores

def main(parser_args):
    """
    Main function for training and validating as model

    Args:
        parser_args (dict): parsed arguments
    """
    open(os.path.join(parser_args.model_save_path, 'log.txt'), 'w').close()
    
    dataloader_train, dataloader_val, wss_mean_std = get_dataloaders(parser_args)
    unet = SphericalUNet(parser_args.pooling_class, parser_args.n_pixels,
                         parser_args.depth, parser_args.laplacian_type,
                         parser_args.kernel_size)
    unet, device = init_device(parser_args.device, unet)
    optimizer = optim.Adam(unet.parameters(), lr=parser_args.learning_rate)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.9, patience=8)
    log('Number of parameters:' + str(sum([p.numel() for p in unet.parameters() if p.requires_grad])), parser_args)
    log('Hyperparameters: ' + str(parser_args), parser_args)
    
    best_metric_score, earlystop_cnt = float('-inf'), 0
    for epoch in range(1, parser_args.n_epochs+1):
        log('Starting Epoch: {}'.format(epoch), parser_args)
        train_losses = train(parser_args, unet, dataloader_train, epoch, optimizer, device)
        _ = validate(parser_args, unet, dataloader_train, epoch, device, name='Train')
        scores = validate(parser_args, unet, dataloader_val, epoch, device, name='Val')
        #scheduler.step(scores['avg_pearson'])
        
        if parser_args.earlystopping_patience <= 0:  # No early stopping
            if not (epoch % 10):
                log('Saving model', parser_args)
                save_checkpoint(unet, epoch, optimizer, wss_mean_std, os.path.join(parser_args.model_save_path, 'unet_best.pt'))
        else:  # Early stopping
            if scores['avg_pearson'] > best_metric_score:
                best_metric_score = scores['avg_pearson']
                earlystop_cnt = 0
                log('Saving best model', parser_args)
                save_checkpoint(unet, epoch, optimizer, wss_mean_std, os.path.join(parser_args.model_save_path, 'unet_best.pt'))
            else:
                earlystop_cnt += 1
                if earlystop_cnt >= parser_args.earlystopping_patience:
                    earlystop_cnt = 0
                    log('Early stopping at epoch %d' % epoch, parser_args)
                    break

if __name__ == "__main__":
    PARSER_ARGS = parse_config(create_parser())
    main(PARSER_ARGS)
