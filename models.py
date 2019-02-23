import time
import math
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import imutil
from logutil import TimeSeries
from tqdm import tqdm
from spatial_recurrent import CSRN
from coordconv import CoordConv2d

INPUT_FRAMES = 3
INPUT_CHANNELS = 3 * INPUT_FRAMES


class Transition(nn.Module):
    def __init__(self, latent_size, num_actions):
        super().__init__()
        # Input: State + Action
        # Output: State
        self.latent_size = latent_size
        self.conv1 = SpectralNorm(nn.Conv2d(latent_size + num_actions, 32, (4,4), stride=2, padding=1))
        self.conv2 = SpectralNorm(nn.Conv2d(32, 64, (4,4), stride=2, padding=1))
        self.conv3 = SpectralNorm(nn.Conv2d(64, 128, (4,4), stride=2, padding=1))

        self.conv4 = SpectralNorm(nn.ConvTranspose2d(128, 64, (4,4), stride=2, padding=1))
        self.conv5 = SpectralNorm(nn.ConvTranspose2d(64, 32, (4,4), stride=2, padding=1))
        self.conv6 = nn.ConvTranspose2d(32, latent_size, (4,4), stride=2, padding=1)
        self.cuda()

    def forward(self, z_map, actions):
        batch_size, z, height, width = z_map.shape
        batch_size_actions, num_actions = actions.shape
        assert batch_size == batch_size_actions

        # First, broadcast the actions across the map
        # Then apply the next-frame prediction CNN
        actions = actions.unsqueeze(-1).unsqueeze(-1)
        actions = actions.repeat(1, 1, height, width)

        x = torch.cat([z_map, actions], dim=1)
        x = self.conv1(x)
        x = F.leaky_relu(x)
        x = self.conv2(x)
        x = F.leaky_relu(x)
        x = self.conv3(x)
        x = F.leaky_relu(x)
        x = self.conv4(x)
        x = F.leaky_relu(x)
        x = self.conv5(x)
        x = F.leaky_relu(x)
        x = self.conv6(x)
        x = torch.sigmoid(x)
        return x


class Encoder(nn.Module):
    def __init__(self, latent_size):
        super().__init__()
        self.latent_size = latent_size
        # Bx1x64x64
        self.conv1 = nn.Conv2d(INPUT_CHANNELS, 64, (5,5), stride=1, padding=2)
        #self.bn_conv1 = nn.BatchNorm2d(32)
        # Bx8x32x32
        self.conv2 = nn.Conv2d(64, 64, (3,3), stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, (5,5), stride=2, padding=2)
        self.conv4 = nn.Conv2d(64, latent_size, (5,5), stride=1, padding=2)

        # Bxlatent_size
        self.cuda()

    def forward(self, x):
        # Input: B x 1 x 64 x 64
        batch_size, frames, channels, height, width = x.shape
        x = x.view(batch_size, frames*channels, height, width)

        x = self.conv1(x)
        x = F.leaky_relu(x)

        x = self.conv2(x)
        x = F.leaky_relu(x)

        x = self.conv3(x)
        x = F.leaky_relu(x)

        x = self.conv4(x)
        x = torch.sigmoid(x)
        return x


from spectral_normalization import SpectralNorm
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        # Bx1x64x64
        self.conv1 = SpectralNorm(nn.Conv2d(3, 32, 1, stride=1, padding=0))
        #self.bn_conv1 = nn.BatchNorm2d(32)
        # Bx8x32x32
        self.conv2 = SpectralNorm(nn.Conv2d(32, 32, 1, stride=1, padding=0))
        #self.bn_conv2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 1, 3, padding=1)

        self.cuda()

    def forward(self, x):
        # Input: B x 1 x 64 x 64
        batch_size, channels, height, width = x.shape

        x = self.conv1(x)
        #x = self.bn_conv1(x)
        x = F.leaky_relu(x)

        x = self.conv2(x)
        #x = self.bn_conv2(x)
        x = F.leaky_relu(x)

        x = self.conv3(x)
        return x.sum(dim=-1).sum(dim=-1).sum(dim=-1)


class RewardPredictor(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(latent_dim, 32, (3,3), stride=1, padding=0)
        self.conv2 = nn.Conv2d(32, 1, (3,3), stride=1, padding=0)
        self.cuda()

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        return x.mean(-1).mean(-1).sum(-1)


class Decoder(nn.Module):
    def __init__(self, latent_size):
        super().__init__()
        self.latent_size = latent_size

        # Bx1x64x64
        self.conv1 = nn.ConvTranspose2d(latent_size, latent_size*4, (4,4),
                        stride=2, padding=1, groups=latent_size, bias=False)
        #self.bn_conv1 = nn.BatchNorm2d(32)
        # Bx8x32x32
        self.conv2 = nn.ConvTranspose2d(latent_size*4, latent_size*3, (3,3),
                        stride=1, padding=1, groups=latent_size, bias=False)
        #self.bg = nn.Parameter(torch.zeros((3, IMG_SIZE, IMG_SIZE)).cuda())
        self.cuda()

    def forward(self, z_map, visualize=False):
        batch_size, latent_size, height, width = z_map.shape

        x = self.conv1(z_map)
        #x = self.bn_conv1(x)
        x = F.leaky_relu(x)

        x = self.conv2(x)
        # Sum the separate items
        x = x.view(batch_size, latent_size, 3, height*2, width*2)

        # Optional: Learn to subtract static background, separate from objects
        #x = x + self.bg
        if visualize:
            visualization = imutil.show(x[0], img_padding=8, save=False, display=False, return_pixels=True)
        x = torch.sum(x, dim=1)
        if visualize:
            return x, visualization
        return x


# https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/8
class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        self.padding = [int(kernel_size / 2)] * dim
        kernel_size = [kernel_size] * dim
        sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / (2 * std)) ** 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )
        self.cuda()

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups, padding=self.padding)


# Normalize a batch of latent points to the unit hypersphere
def norm(x):
    norm = torch.norm(x, p=2, dim=1)
    x = x / (norm.expand(1, -1).t() + .0001)
    return x


