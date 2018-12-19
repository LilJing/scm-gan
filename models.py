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


class Transition(nn.Module):
    def __init__(self, latent_size, num_actions):
        super().__init__()
        # Input: State + Action
        # Output: State
        self.latent_size = latent_size
        self.fc1 = nn.Linear(num_actions + self.latent_size, 128)
        self.fc2 = nn.Linear(128, latent_size)
        self.cuda()

    def forward(self, z, actions):
        x = torch.cat([z, actions], dim=1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return z + x


class Encoder(nn.Module):
    def __init__(self, latent_size):
        super().__init__()
        self.latent_size = latent_size
        # Bx1x64x64
        self.conv1 = CoordConv2d(3 + 2, 32, 4, stride=2, padding=1)
        self.bn_conv1 = nn.BatchNorm2d(32)
        # Bx8x32x32
        self.conv2 = CoordConv2d(32 + 2, 32, 4, stride=2, padding=1)
        self.bn_conv2 = nn.BatchNorm2d(32)

        self.fc1 = nn.Linear(32*16*16, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, latent_size)
        self.fc2.weight.data.normal_(0, .01)

        # Bxlatent_size
        self.cuda()

    def forward(self, x, visual_tag=None):
        # Input: B x 1 x 64 x 64
        batch_size, channels, height, width = x.shape

        x = self.conv1(x)
        x = self.bn_conv1(x)
        x = F.leaky_relu(x)

        x = self.conv2(x)
        x = self.bn_conv2(x)
        x = F.leaky_relu(x)

        x = x.view(-1, 32*16*16)
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x)

        x = self.fc2(x)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        # Bx1x64x64
        self.conv1 = nn.Conv2d(3, 32, 1, stride=1, padding=0)
        self.bn_conv1 = nn.BatchNorm2d(32)
        # Bx8x32x32
        self.conv2 = nn.Conv2d(32, 32, 1, stride=1, padding=0)
        self.bn_conv2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 1, 3, padding=1)

        self.cuda()

    def forward(self, x, visual_tag=None):
        # Input: B x 1 x 64 x 64
        batch_size, channels, height, width = x.shape

        x = self.conv1(x)
        x = self.bn_conv1(x)
        x = F.leaky_relu(x)

        x = self.conv2(x)
        x = self.bn_conv2(x)
        x = F.leaky_relu(x)

        x = self.conv3(x)
        return x.sum(dim=-1).sum(dim=-1).sum(dim=-1)


class Decoder(nn.Module):
    def __init__(self, latent_size, k=64, m=12):
        super().__init__()
        self.latent_size = latent_size
        self.width = 64
        num_places = latent_size // 4
        # K is the number of buckets to use when discretizing
        self.k = k
        # M is the number of kinds of things that there can be
        #self.m = m

        self.to_categorical = RealToCategorical(z=latent_size//2, k=k)

        global_channels = num_places*4

        # Separable convolutions
        self.pad_conv1 = nn.ReflectionPad2d(2)
        #self.places_conv1 = nn.Conv2d(num_places, num_places*16, kernel_size=5, groups=num_places, bias=False)
        #self.pad_conv2 = nn.ReflectionPad2d(2)
        #self.places_conv2 = nn.Conv2d(num_places*16, num_places*16, kernel_size=5, groups=num_places, bias=False)
        self.to_rgb = nn.Conv2d(num_places* (1 + global_channels), num_places*3, kernel_size=5, groups=num_places, bias=False)

        # Test: just RGB
        #self.to_rgb = nn.ConvTranspose2d(num_places, num_places*3, groups=num_places, kernel_size=5, padding=2, bias=False)
        #self.to_rgb.weight.data = torch.abs(self.to_rgb.weight.data)


        self.things_fc1 = nn.Linear(self.latent_size//2, global_channels)
        #self.spatial_sample = SpatialSampler(k=k)
        self.spatial_map = SpatialCoordToMap(k=k, z=latent_size)

        self.cuda()

    def forward(self, z, visual_tag=None, enable_aux_loss=False):
        # The world consists of things in places.
        batch_size = len(z)
        num_places = self.latent_size // 4
        z_places, z_things = z[:, :num_places*2], z[:, num_places*2:]

        # Compute places as ring of outer products, one place per latent dim
        x_cat = self.to_categorical(z_places)
        if visual_tag:
            imutil.show(x_cat[0], font_size=8, resize_to=(self.k, self.latent_size),
                        filename='visual_to_cat_{}.png'.format(visual_tag))

        # Failed experiment: Sample stochastically from the distribution!
        #places, sampled_points = self.spatial_sample(x_cat)
        #sample_from = sampled_points / (sampled_points.sum(dim=3, keepdim=True).sum(dim=2, keepdim=True))

        places = self.spatial_map(x_cat)

        # Hack: disincentivize overlap among position distributions
        aux_loss = torch.mean(places.sum(dim=1)**2)
        #places = places / (places.mean() + places.sum(dim=1, keepdim=True))

        if visual_tag:
            cap = 'Places min {:.03f} max {:.03f}'.format(places.min(), places.max())
            imutil.show(places[0] / places[0].max(), filename='visual_places_{}.png'.format(visual_tag),
                        resize_to=(512, 512), caption=cap, img_padding=8)


        # Append non-location-specific information to each "place" channel
        x_things = self.things_fc1(z_things)
        x_things = x_things.unsqueeze(1).unsqueeze(3).unsqueeze(4)
        x_things = x_things.repeat(1, num_places, 1, self.width, self.width)

        # Apply separable convolutions to draw one "thing" at each sampled location
        x_places = places.unsqueeze(2)
        x = torch.cat([x_places, x_things], dim=2)
        x = x.view(batch_size, -1, self.width, self.width)

        #x = F.leaky_relu(x, 0.2)
        #x = self.pad_conv2(x)
        #x = self.places_conv2(x)
        #x = F.leaky_relu(x, 0.2)
        x = self.pad_conv1(x)
        x = self.to_rgb(x)
        x = x.view(batch_size, num_places, 3, 64, 64)
        if visual_tag:
            cap = 'Things min {:.03f} max {:.03f}'.format(x.min(), x.max())
            img = x[0] - x[0].min()
            imutil.show(img, filename='the_things_{}.png'.format(visual_tag),
                        resize_to=(512, 1024), caption=cap, font_size=8, img_padding=10)

        # Combine independent additive objects-in-locations
        x = x.sum(dim=1)

        if enable_aux_loss:
            # Hack: add noise
            #x += x.new(x.shape).normal_(0, .01)
            return x, aux_loss
        x = torch.tanh(x)
        return x


# Input: categorical estimates of x and y coordinates
# Output: 2d feature maps with all but one pixel masked to zero
class SpatialSampler(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k
        self.t = 0

    def forward(self, x_cat):
        batch_size, num_axes, k = x_cat.shape
        num_places = num_axes // 2
        self.places = torch.zeros((batch_size, num_places, self.k, self.k)).cuda()
        self.sampled_points = torch.zeros((batch_size, num_places, self.k, self.k)).cuda()
        for i in range(0, num_places):
            horiz, vert = x_cat[:,i*2], x_cat[:,(i*2)+1]
            self.places[:, i] = torch.einsum('ij,ik->ijk', [horiz, vert])
            # Sample from horiz and from vert to select a spatial point
            beta = .1 + .1 * np.sin(self.t / 1000)
            horiz = gumbel_sample_1d(horiz, beta=beta)
            vert = gumbel_sample_1d(vert, beta=beta)
            self.sampled_points[:, i] = torch.einsum('ij,ik->ijk', [horiz, vert])
        self.sampled_points *= 100
        self.t += 1
        return self.places, self.sampled_points


class SpatialCoordToMap(nn.Module):
    def __init__(self, k, z):
        super().__init__()
        self.k = k
        self.z = z
        self.num_places = z // 4
        self.gamma = nn.Parameter(torch.ones(1).cuda())
        self.alpha = torch.nn.Parameter(torch.zeros(self.num_places).cuda())

    def forward(self, x_cat):
        batch_size, num_axes, k = x_cat.shape
        num_places = num_axes // 2
        places = torch.zeros((batch_size, num_places, self.k, self.k)).cuda()
        for i in range(0, num_places):
            horiz, vert = x_cat[:,i*2], x_cat[:,(i*2)+1]
            places[:, i] = torch.einsum('ij,ik->ijk', [horiz, vert])
        dots = (places == places.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0].cuda()).float()
        places = places + torch.einsum('bchw,c->bchw', [dots, torch.exp(self.alpha)])
        return places


def gumbel_sample_1d(pdf, beta=1.0):
    from torch.distributions.gumbel import Gumbel
    noise = Gumbel(torch.zeros(size=pdf.shape), beta * torch.ones(size=pdf.shape)).sample().cuda()
    log_pdf = torch.log(pdf) + noise
    max_points = log_pdf.max(dim=1, keepdim=True)[0]
    return pdf * (log_pdf == max_points).type(torch.FloatTensor).cuda()


def gumbel_sample_2d(pdf, beta=1.0):
    batch_size, channels, height, width = pdf.shape
    # Generate a density function (differentiable)
    pdf_probs = pdf / pdf.sum(dim=2, keepdim=True).sum(dim=3, keepdim=True)

    # Perform sampling (non-differentiable)
    log_probs = F.log_softmax(pdf_probs.view(batch_size, channels, -1), dim=2).view(pdf.shape)
    from torch.distributions.gumbel import Gumbel
    beta = 1 / (height*width)
    noise = Gumbel(torch.zeros(size=pdf.shape), beta * torch.ones(size=pdf.shape)).sample().cuda()
    log_probs += noise
    max_values = log_probs.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
    sampled_mask = (log_probs == max_values).type(torch.FloatTensor).cuda()

    # Mask out all but the sampled values
    pdf = pdf_probs * sampled_mask
    return pdf


class RealToCategorical(nn.Module):
    """
    Input: Real values eg. -0.25, 4.1, 0
    Output: Categorical encoding like:
        00010000000000
        00000000000001
        00000001000000
    OR Thermometer encoding like:
        11110000000000
        11111111111111
        11111111000000
    """
    def __init__(self, z, k, kernel='gaussian'):
        super().__init__()
        self.z = z
        self.k = k
        self.kernel = kernel
        self.rho = torch.arange(-1, 1, 2/k).unsqueeze(0).repeat(z, 1).cuda()
        # For learnable particle positions
        #self.rho = torch.nn.Parameter(self.rho)

        # Sharpness/scale parameter
        self.eta = torch.nn.Parameter(torch.zeros(self.z).cuda())
        self.gamma = torch.nn.Parameter(torch.zeros(self.z).cuda())
        self.beta = torch.nn.Parameter(torch.zeros(self.z).cuda())
        self.register_backward_hook(self.clamp_weights)
        self.cuda()


    def forward(self, x):
        # x is a real-valued tensor size (batch, Z)
        batch_size = len(x)
        x = torch.tanh(x)
        # Broadcast x to (batch, Z, K)
        perceived_locations = x.unsqueeze(-1).repeat(1, 1, self.k)
        reference_locations = self.rho.unsqueeze(0).repeat(batch_size, 1, 1)
        distances = (perceived_locations - reference_locations)
        distances = torch.einsum('bzk,z->bzk', [distances, torch.exp(self.eta) + 1])
        # IMQ kernel
        if self.kernel == 'inverse_multiquadratic':
            eps = .1
            kern = eps / (eps + distances**2)
        elif self.kernel == 'gaussian':
            kern = torch.exp(-distances**2)
        kern = torch.einsum('bzk,z->bzk', [kern, torch.exp(self.gamma)])

        # Output is a category between 1 and K, for each of the Z real values
        probs = kern

        # Hack: Normalize kernel without running softmax
        #probs = kern / kern.sum(dim=2, keepdim=True)

        # Hack: Run softmax, at multiple scales
        #probs = torch.softmax(kern, dim=2) + torch.softmax(kern*1000, dim=2)

        # Hack: make the center pixel stand out (per dimension)
        #dots = (probs == probs.max(dim=2, keepdim=True)[0]).float()
        #probs = probs + torch.einsum('bzk,z->bzk', [dots, torch.exp(self.beta)])
        return probs

    def clamp_weights(self, *args):
        self.eta.data.clamp_(min=-2, max=+2)


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
