"""
Encoder for Spherical UNet.
"""
from torch import nn

from layers.chebyshev import SphericalChebConv
import torch.nn.functional as F
from models.spherical_unet.utils import SphericalChebBN, SphericalChebBNPool


class SphericalChebBN2(nn.Module):
    """
    Building Block made of 2 Building Blocks (convolution, batchnorm, activation).
    """

    def __init__(self, in_channels, middle_channels, out_channels, lap, kernel_size):
        """
        Initialization.

        Args:
            in_channels (int): initial number of channels.
            middle_channels (int): middle number of channels.
            out_channels (int): output number of channels.
            lap (:obj:`torch.sparse.FloatTensor`): laplacian.
            kernel_size (int, optional): polynomial degree.
        """

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spherical_cheb_bn_1 = SphericalChebBN(in_channels, middle_channels, lap, kernel_size)
        self.spherical_cheb_bn_2 = SphericalChebBN(middle_channels, out_channels, lap, kernel_size)

    def forward(self, x):
        """
        Forward Pass.

        Args:
            x (:obj:`torch.Tensor`): input [batch x vertices x channels/features]

        Returns:
            :obj:`torch.Tensor`: output [batch x vertices x channels/features]
        """
        x = self.spherical_cheb_bn_1(x)
        x = self.spherical_cheb_bn_2(x)
        return x


class SphericalChebPool(nn.Module):
    """
    Building Block with a pooling/unpooling and a Chebyshev Convolution.
    """

    def __init__(self, in_channels, out_channels, lap, pooling, kernel_size):
        """
        Initialization.

        Args:
            in_channels (int): initial number of channels.
            out_channels (int): output number of channels.
            lap (:obj:`torch.sparse.FloatTensor`): laplacian.
            pooling (:obj:`torch.nn.Module`): pooling/unpooling module.
            kernel_size (int, optional): polynomial degree.
        """
        super().__init__()
        self.pooling = pooling
        self.spherical_cheb = SphericalChebConv(in_channels, out_channels, lap, kernel_size)
        self.act = nn.ELU()
        self.batchnorm = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        """
        Forward Pass.

        Args:
            x (:obj:`torch.Tensor`): input [batch x vertices x channels/features]

        Returns:
            :obj:`torch.Tensor`: output [batch x vertices x channels/features]
        """
        x = self.pooling(x)
        x = self.spherical_cheb(x)
        x = self.batchnorm(x.permute(0, 2, 1))
        x = self.act(x.permute(0, 2, 1))
        x = self.dropout(x)
        return x


class Encoder(nn.Module):
    """
    Encoder for the Spherical UNet.
    """

    def __init__(self, pooling, laps, kernel_size):
        """
        Initialization.

        Args:
            pooling (:obj:`torch.nn.Module`): pooling layer.
            laps (list): List of laplacians.
            kernel_size (int): polynomial degree.
        """
        super().__init__()
        self.pooling = pooling
        self.kernel_size = kernel_size
        self.enc_l3 = SphericalChebBN2(1, 32, 64, laps[3], self.kernel_size)
        self.pool2 = SphericalChebBNPool(64, 128, laps[2], self.pooling, self.kernel_size)
        self.pool1 = SphericalChebBNPool(128, 256, laps[1], self.pooling, self.kernel_size)
        self.pool0 = SphericalChebBNPool(256, 512, laps[0], self.pooling, self.kernel_size)

    def forward(self, x):
        """
        Forward Pass.

        Args:
            x (:obj:`torch.Tensor`): input [batch x vertices x channels/features]

        Returns:
            x_enc* :obj: `torch.Tensor`: output [batch x vertices x channels/features]
        """
        x_enc3 = self.enc_l3(x)
        x_enc2 = self.pool2(x_enc3)
        x_enc1 = self.pool1(x_enc2)
        x_enc0 = self.pool0(x_enc1)

        return x_enc0, x_enc1, x_enc2, x_enc3
