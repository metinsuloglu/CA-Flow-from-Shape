"""
Decoder for Spherical UNet.
"""

import torch
from torch import nn
import torch.nn.functional as F

from layers.chebyshev import SphericalChebConv
from models.spherical_unet.utils import SphericalChebBN, SphericalChebBNPool


class SphericalChebBNPoolCheb(nn.Module):
    """
    Building Block calling a SphericalChebBNPool block then a SphericalCheb.
    """

    def __init__(self, in_channels, middle_channels, out_channels, lap, pooling, kernel_size):
        """
        Initialization.

        Args:
            in_channels (int): initial number of channels.
            middle_channels (int): middle number of channels.
            out_channels (int): output number of channels.
            lap (:obj:`torch.sparse.FloatTensor`): laplacian.
            pooling (:obj:`torch.nn.Module`): pooling/unpooling module.
            kernel_size (int, optional): polynomial degree. Defaults to 3.
        """
        super().__init__()
        self.spherical_cheb_bn_pool = SphericalChebBNPool(in_channels, middle_channels, lap, pooling, kernel_size)
        self.spherical_cheb = SphericalChebConv(middle_channels, out_channels, lap, kernel_size)

    def forward(self, x):
        """
        Forward Pass.

        Args:
            x (:obj:`torch.Tensor`): input [batch x vertices x channels/features]

        Returns:
            :obj:`torch.Tensor`: output [batch x vertices x channels/features]
        """
        x = self.spherical_cheb_bn_pool(x)
        x = self.spherical_cheb(x)
        return x


class SphericalChebBNPoolConcat(nn.Module):
    """
    Building Block calling a SphericalChebBNPool Block
    then concatenating the output with another tensor
    and calling a SphericalChebBN block.
    """

    def __init__(self, in_channels, out_channels, lap, pooling, kernel_size):
        """
        Initialization.

        Args:
            in_channels (int): initial number of channels.
            out_channels (int): output number of channels.
            lap (:obj:`torch.sparse.FloatTensor`): laplacian.
            pooling (:obj:`torch.nn.Module`): pooling/unpooling module.
            kernel_size (int, optional): polynomial degree. Defaults to 3.
        """
        super().__init__()
        self.spherical_cheb_bn_pool = SphericalChebBNPool(in_channels, out_channels, lap, pooling, kernel_size)
        self.spherical_cheb_bn = SphericalChebBN(2*out_channels, out_channels, lap, kernel_size)

    def forward(self, x, concat_data):
        """
        Forward Pass.

        Args:
            x (:obj:`torch.Tensor`): input [batch x vertices x channels/features]
            concat_data (:obj:`torch.Tensor`): encoder layer output [batch x vertices x channels/features]

        Returns:
            :obj:`torch.Tensor`: output [batch x vertices x channels/features]
        """
        x = self.spherical_cheb_bn_pool(x)
        x = torch.cat((x, concat_data), dim=2)
        x = self.spherical_cheb_bn(x)
        return x


class Decoder(nn.Module):
    """
    The decoder of the Spherical UNet.
    """

    def __init__(self, unpooling, laps, kernel_size):
        """
        Initialization.

        Args:
            unpooling (:obj:`torch.nn.Module`): The unpooling object.
            laps (list): List of laplacians.
        """
        super().__init__()
        self.unpooling = unpooling
        self.kernel_size = kernel_size
        self.dec_l1 = SphericalChebBNPoolConcat(512, 256, laps[1], self.unpooling, self.kernel_size)
        self.dec_l2 = SphericalChebBNPoolConcat(256, 128, laps[2], self.unpooling, self.kernel_size)
        self.dec_l3 = SphericalChebBNPoolConcat(128, 64, laps[3], self.unpooling, self.kernel_size)
        self.conv_1 = nn.Conv1d(64, 32, 1)
        self.conv_2 = nn.Conv1d(32, 1, 1)
        self.batchnorm = nn.BatchNorm1d(32)
        self.dropout = nn.Dropout(p=0.1)
        self.act = nn.ELU()

    def forward(self, x_enc0, x_enc1, x_enc2, x_enc3):
        """
        Forward Pass.

        Args:
            x_enc* (:obj:`torch.Tensor`): input tensors.

        Returns:
            :obj:`torch.Tensor`: output after forward pass.
        """
        x = self.dec_l1(x_enc0, x_enc1)
        x = self.dec_l2(x, x_enc2)
        x = self.dec_l3(x, x_enc3)
        x = self.conv_1(x.permute(0, 2, 1))
        x = self.batchnorm(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.conv_2(x)
        return x.permute(0, 2, 1)
