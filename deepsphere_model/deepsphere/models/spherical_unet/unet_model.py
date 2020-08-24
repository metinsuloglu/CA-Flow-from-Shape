"""
Spherical Graph Convolutional Neural Network with UNet autoencoder architecture.
"""

import torch
from torch import nn

from layers.samplings.equiangular_pool_unpool import Equiangular
from layers.samplings.healpix_pool_unpool import Healpix
from layers.samplings.icosahedron_pool_unpool import Icosahedron
from models.spherical_unet.decoder import Decoder
from models.spherical_unet.encoder import Encoder
from utils.laplacian_funcs import get_equiangular_laplacians, get_healpix_laplacians, get_icosahedron_laplacians


class SphericalUNet(nn.Module):

    def __init__(self, pooling_class, N, depth, laplacian_type, kernel_size, ratio=1):
        """
        Initialization.

        Args:
            pooling_class (obj): One of three classes of pooling methods
            N (int): Number of pixels in the input image
            depth (int): The depth of the UNet, which is bounded by the N and the type of pooling
            kernel_size (int): chebychev polynomial degree
            ratio (float): Parameter for equiangular sampling
        """
        super().__init__()
        self.ratio = ratio
        self.kernel_size = kernel_size
        if pooling_class == "icosahedron":
            self.pooling_class = Icosahedron()
            self.laps = get_icosahedron_laplacians(N, depth, laplacian_type)
        elif pooling_class == "healpix":
            self.pooling_class = Healpix()
            self.laps = get_healpix_laplacians(N, depth, laplacian_type)
        elif pooling_class == "equiangular":
            self.pooling_class = Equiangular()
            self.laps = get_equiangular_laplacians(N, depth, self.ratio, laplacian_type)
        else:
            raise ValueError("Error: sampling method unknown. Please use icosahedron, healpix or equiangular.")

        self.encoder = Encoder(self.pooling_class.pooling, self.laps, self.kernel_size)
        self.decoder = Decoder(self.pooling_class.unpooling, self.laps, self.kernel_size)

    def forward(self, x):
        """
        Forward Pass.

        Args:
            x (:obj:`torch.Tensor`): input to be forwarded.

        Returns:
            :obj:`torch.Tensor`: output
        """
        x_encoder = self.encoder(x)
        output = self.decoder(*x_encoder)
        return output
