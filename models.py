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
from spectral_normalization import SpectralNorm

NOISE_DIM = 3
ENCODER_INPUT_FRAMES = 3

#ts = TimeSeries('Profiling')


def random_eps(p=0.5, batch_size=32, height=64, width=64, channels=NOISE_DIM):
    shape = (batch_size, height, width, channels)
    return torch.bernoulli(torch.ones(shape) * p).cuda()


from torch.autograd import Function
class DifferentiableBernoulliSampler(Function):
    @staticmethod
    def forward(ctx, x):
        # In the forward pass, discretize by sampling
        #ctx.save_for_backward(x)
        return torch.bernoulli(x)

    @staticmethod
    def backward(ctx, grad_output):
        # In the backward pass, change nothing
        return grad_output


class Transition(nn.Module):
    def __init__(self, latent_size, num_actions):
        super().__init__()
        # Input: State + Action
        # Output: State
        self.latent_size = latent_size

        # Skip connections from output of 1 to input of 6, and output of 2 to input of 5
        self.conv1 = SpectralNorm(nn.Conv2d(latent_size + num_actions, 16, (4,4), stride=2, padding=1))
        self.conv2 = SpectralNorm(nn.Conv2d(16, 32, (4,4), stride=2, padding=1))
        #self.conv3 = (nn.Conv2d(32, 64, (4,4), stride=2, padding=1))
        #self.conv4 = (nn.ConvTranspose2d(64, 32, (4,4), stride=2, padding=1))
        self.conv5 = SpectralNorm(nn.ConvTranspose2d(32, 16, (4,4), stride=2, padding=1))
        self.conv6 = nn.ConvTranspose2d(16 + 16, latent_size, (4,4), stride=2, padding=1)
        self.cuda()

    def forward(self, s, a, eps=None):
        start_time = time.time()

        actions = a
        z_map = s
        batch_size, z, height, width = z_map.shape
        batch_size_actions, num_actions = actions.shape
        assert batch_size == batch_size_actions

        #if eps is None:
        #    eps = random_eps(batch_size=batch_size)
        #assert len(eps) == batch_size

        # Broadcast the actions across the convolutional map
        actions = actions.unsqueeze(-1).unsqueeze(-1)
        actions = actions.repeat(1, 1, height, width)

        # Stack the latent values, the actions, and random chance
        x = torch.cat([z_map, actions], dim=1)

        # Convolve down, saving skip activations like U-net
        x = self.conv1(x)
        x = F.leaky_relu(x)
        skip1 = x.clone()

        x = self.conv2(x)
        x = F.leaky_relu(x)

        #skip2 = x.clone()

        #x = self.conv3(x)
        #x = F.leaky_relu(x)

        # Convolve back up, using saved skips
        #x = self.conv4(x)
        #x = F.leaky_relu(x)

        #x = torch.cat([x, skip2], dim=1)
        x = self.conv5(x)
        x = F.leaky_relu(x)

        x = torch.cat([x, skip1], dim=1)
        x = self.conv6(x)
        x = torch.sigmoid(x)

        # And now, to make it stochastic: sample from the resulting
        # factorized multivariate Bernoulli distribution
        if self.training:
            #x = DifferentiableBernoulliSampler.apply(x)
            pass
        else:
            # During test time, drop the sampling
            #x = (x > 0.5).type(x.type())
            #x = DifferentiableBernoulliSampler.apply(x)
            pass

        #ts.collect('Transition', time.time() - start_time)
        #ts.print_every(10)
        return x



class Encoder(nn.Module):
    def __init__(self, latent_size, color_channels):
        super().__init__()
        self.latent_size = latent_size
        self.color_channels = color_channels
        # Bx1x64x64
        self.conv1 = SpectralNorm(nn.Conv2d(color_channels * ENCODER_INPUT_FRAMES, 64, (3,3), stride=1, padding=1))
        #self.bn_conv1 = nn.BatchNorm2d(32)
        # Bx8x32x32
        #self.conv2 = SpectralNorm(nn.Conv2d(64, 64, (3,3), stride=1, padding=1))
        #self.conv3 = SpectralNorm(nn.Conv2d(64, 64, (3,3), stride=2, padding=1))
        self.conv4 = nn.Conv2d(64, latent_size, (4,4), stride=2, padding=1)

        # Bxlatent_size
        self.cuda()

    def forward(self, x):
        # Input: B x 1 x 64 x 64
        start_time = time.time()
        batch_size, frames, channels, height, width = x.shape
        x = x.view(batch_size, frames*channels, height, width)

        x = self.conv1(x)
        x = F.leaky_relu(x)

        #x = self.conv2(x)
        #x = F.leaky_relu(x)

        #x = self.conv3(x)
        #x = F.leaky_relu(x)

        x = self.conv4(x)
        x = torch.sigmoid(x)
        #ts.collect('Encoder', time.time() - start_time)
        return x


# The world is deterministic and completely predictable
# However, one of the inputs to the world is a map of random values
# The random values have two properties:
#     1. They are impossible to guess beforehand
#     2. It is obvious what their value was, after the fact
# The discriminator solves #1 and this network solves #2
#
class Inverter(nn.Module):
    def __init__(self, latent_size):
        super().__init__()
        self.latent_size = latent_size
        self.conv1 = nn.Conv2d(latent_size * 2, 32, (3,3), stride=1, padding=1)
        self.conv2 = SpectralNorm(nn.Conv2d(32, NOISE_DIM, (3,3), stride=1, padding=0))

        # Bxlatent_size
        self.cuda()

    # Given s_{t-1}, s_t, a_{t}, infer \epsilon_{t-1}
    def forward(self, s_curr, s_next, a):
        # Input: B x 1 x 64 x 64
        start_time = time.time()
        batch_size, frames, channels, height, width = x.shape
        x = x.view(batch_size, frames*channels, height, width)

        x = self.conv1(x)
        x = F.leaky_relu(x)

        x = self.conv2(x)
        x = torch.sigmoid(x)
        #ts.collect('NoiseRecognizer', time.time() - start_time)
        return x


# Input: A noise map, either output by the NoiseRecognizer or drawn from the noise prior
# Output: Linear unit for a binary classification, random or not random
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        # Bx1x64x64
        self.conv1 = SpectralNorm(nn.Conv2d(NOISE_DIM, 32, (3, 3), stride=2, padding=0))
        # Bx32x32x32
        self.conv2 = SpectralNorm(nn.Conv2d(32, 32, (3, 3), stride=2, padding=0))
        # Bx32x16x16
        self.conv3 = nn.Conv2d(32, 32, (3,3), stride=2, padding=0)

        self.fc1 = nn.Linear(32*7*7, 1)
        self.cuda()

    def forward(self, x):
        # Input: B x 1 x 64 x 64
        batch_size, channels, height, width = x.shape

        x = self.conv1(x)
        x = F.leaky_relu(x)

        x = self.conv2(x)
        x = F.leaky_relu(x)

        x = self.conv3(x)
        x = F.leaky_relu(x)

        x = self.fc1(x)
        x = F.leaky_relu(x)
        return x


class RewardPredictor(nn.Module):
    # Predicts multiple reward types, if you have multiple reward signals
    def __init__(self, latent_dim, num_rewards):
        super().__init__()
        self.conv1 = nn.Conv2d(latent_dim, 32, (3,3), stride=1, padding=0)
        # Each reward is discretized into a 3-way classification: +1, -1, or 0
        self.conv2 = nn.Conv2d(32, num_rewards * 3, (3,3), stride=2, padding=0)
        self.cuda()

    def forward(self, x, visualize=False):
        start_time = time.time()
        x = self.conv1(x)
        x = F.leaky_relu(x)
        x = self.conv2(x)

        # Classify each pixel as +1, -1, or 0 (for each reward type)
        batch_size, channels, height, width = x.shape
        x = x.view(batch_size, 3, channels // 3, height, width)
        x = torch.softmax(x, dim=1)
        # Return the cumulative reward (for each reward type)
        x = x[:, 0] - x[:, 2]
        #ts.collect('RPred', time.time() - start_time)
        if visualize:
            return x.sum(-1).sum(-1), x
        return x.sum(-1).sum(-1)


class Decoder(nn.Module):
    def __init__(self, latent_size, color_channels):
        super().__init__()
        self.latent_size = latent_size
        self.color_channels = color_channels

        # Bx1x64x64
        self.conv1 = nn.ConvTranspose2d(latent_size, latent_size*4, (3,3),
                        stride=2, padding=1, groups=latent_size, bias=False)
        #self.bn_conv1 = nn.BatchNorm2d(32)
        # Bx8x32x32
        self.conv2 = nn.ConvTranspose2d(latent_size * 4,
                                        latent_size*self.color_channels, (4,4),
                                        stride=1, padding=1,
                                        groups=latent_size, bias=False)
        #self.bg = nn.Parameter(torch.zeros((3, IMG_SIZE, IMG_SIZE)).cuda())
        self.cuda()

    def forward(self, z_map, visualize=False):
        start_time = time.time()
        batch_size, latent_size, height, width = z_map.shape

        x = self.conv1(z_map)
        #x = self.bn_conv1(x)
        x = F.leaky_relu(x)

        x = self.conv2(x)
        # Sum the separate items
        x = x.view(batch_size, latent_size, self.color_channels, height*2, width*2)

        # Optional: Learn to subtract static background, separate from objects
        #x = x + self.bg
        if visualize:
            visualization = x[0]
            #imutil.show(x[0], img_padding=8, save=False, display=False, return_pixels=True)
        x = torch.sum(x, dim=1)
        #ts.collect('Decoder', time.time() - start_time)
        if visualize:
            return x, visualization
        return x


class RGBDecoder(nn.Module):
    def __init__(self, color_channels=3, img_size=256):
        super().__init__()
        #self.conv1 = nn.ConvTranspose2d(color_channels, 32, (4,4), stride=2, padding=1)
        #self.conv2 = nn.ConvTranspose2d(32, 3, (4,4), stride=2, padding=1)
        self.bg = nn.Parameter(torch.zeros((color_channels, img_size, img_size)).cuda())
        #self.cuda()

    def forward(self, x, enable_bg=True):
        #x = self.conv1(x)
        #x = F.leaky_relu(x)
        #x = self.conv2(x)
        #if enable_bg:
        #    x = x + self.bg
        #x = torch.sigmoid(x)
        #return x
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


