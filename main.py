import time
import math
import os
import sys
if len(sys.argv) < 2:
    print('Usage: {} datasource'.format(sys.argv[0]))
    print('\tAvailable datasources: boxes, minipong, mediumpong...')
    exit(1)

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

from higgins import higgins_metric

from importlib import import_module
datasource = import_module('envs.' + sys.argv[1])


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
        #self.fc2.weight.data.normal_(0, .1)

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


class Decoder(nn.Module):
    def __init__(self, latent_size, k=64, m=12):
        super().__init__()
        self.latent_size = latent_size
        num_places = latent_size // 4
        # K is the number of buckets to use when discretizing
        self.k = k
        # M is the number of kinds of things that there can be
        self.m = m

        self.to_categorical = RealToCategorical(z=latent_size//2, k=k)

        # Separable convolutions
        self.pad_conv1 = nn.ReflectionPad2d(2)
        self.places_conv1 = nn.Conv2d(num_places, num_places*16, kernel_size=5, groups=num_places, bias=False)
        self.pad_conv2 = nn.ReflectionPad2d(2)
        self.places_conv2 = nn.Conv2d(num_places*16, num_places*16, kernel_size=5, groups=num_places, bias=False)
        self.to_rgb = nn.ConvTranspose2d(num_places*16, num_places*3, groups=num_places, kernel_size=3, padding=1, bias=False)

        # Test: just RGB
        #self.to_rgb = nn.ConvTranspose2d(num_places, num_places*3, groups=num_places, kernel_size=5, padding=2, bias=False)
        #self.to_rgb.weight.data = torch.abs(self.to_rgb.weight.data)

        self.things_fc1 = nn.Linear(self.latent_size//2, num_places*16)

        #self.spatial_sample = SpatialSampler(k=k)
        self.spatial_sample = SpatialCoordToMap(k=k, z=latent_size)

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

        #places, sampled_points = self.spatial_sample(x_cat)
        #sample_from = sampled_points / (sampled_points.sum(dim=3, keepdim=True).sum(dim=2, keepdim=True))
        places = self.spatial_sample(x_cat)

        # Disincentivize overlap among position distributions
        overlap_metric = torch.exp(places.sum(dim=1)).mean()
        #places = places / (places.mean() + places.sum(dim=1, keepdim=True))

        if visual_tag:
            cap = 'Places min {:.03f} max {:.03f}'.format(places.min(), places.max())
            imutil.show(places[0] / places[0].max(), filename='visual_places_{}.png'.format(visual_tag),
                        resize_to=(512, 512), caption=cap, img_padding=8)

        # Apply separable convolutions to draw one "thing" at each sampled location
        x = places
        x = self.pad_conv1(x)
        x = self.places_conv1(x)

        # Append non-location-specific information
        zx = self.things_fc1(z_things)
        zx = torch.tanh(zx)
        x = x * zx.unsqueeze(2).unsqueeze(3)

        x = F.leaky_relu(x, 0.2)
        x = self.pad_conv2(x)
        x = self.places_conv2(x)
        x = F.leaky_relu(x, 0.2)
        x = self.to_rgb(x)
        x = x.view(batch_size, num_places, 3, 64, 64)
        if visual_tag:
            cap = 'Things min {:.03f} max {:.03f}'.format(x.min(), x.max())
            imutil.show(x[0], filename='the_things_{}.png'.format(visual_tag),
                        resize_to=(512,1024), caption=cap, font_size=8, img_padding=10)

        # Combine independent additive objects-in-locations
        x = x.sum(dim=1)

        if enable_aux_loss:
            return x, overlap_metric
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

    def forward(self, x_cat):
        batch_size, num_axes, k = x_cat.shape
        num_places = num_axes // 2
        places = torch.zeros((batch_size, num_places, self.k, self.k)).cuda()
        for i in range(0, num_places):
            horiz, vert = x_cat[:,i*2], x_cat[:,(i*2)+1]
            places[:, i] = torch.einsum('ij,ik->ijk', [horiz, vert])
        #places = places ** self.gamma
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
    def __init__(self, z, k, kernel='inverse_multiquadratic'):
        super().__init__()
        self.z = z
        self.k = k
        self.kernel = kernel
        self.rho = torch.arange(-1, 1, 2/k).unsqueeze(0).repeat(z, 1).cuda()
        # For learnable particle positions
        #self.rho = torch.nn.Parameter(self.rho)

        # Sharpness/scale parameter
        self.eta = torch.nn.Parameter(torch.zeros(self.z).cuda())
        self.gamma = torch.nn.Parameter(torch.ones(self.z).cuda())
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
        distances = torch.einsum('bzk,z->bzk', [distances, torch.exp(self.eta)])
        # IMQ kernel
        if self.kernel == 'inverse_multiquadratic':
            eps = .1
            kern = eps / (eps + distances**2)
        elif self.kernel == 'gaussian':
            kern = torch.exp(-distances**2)
        #kern = torch.einsum('bzk,z->bzk', [kern, self.gamma])
        #kern = kern * 8
        # Output is a category between 1 and K, for each of the Z real values
        #probs = kern / kern.sum(dim=2, keepdim=True)
        probs = torch.softmax(kern, dim=2) + torch.softmax(kern*1000, dim=2)
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


# Inverse multiquadratic kernel with varying kernel bandwidth
# Tolstikhin et al. https://arxiv.org/abs/1711.01558
# https://github.com/schelotto/Wasserstein_Autoencoders
def imq_kernel(X: torch.Tensor,
               Y: torch.Tensor,
               h_dim: int):
    batch_size = X.size(0)

    p2_norm_x = X.pow(2).sum(1).unsqueeze(0)
    norms_x = X.sum(1).unsqueeze(0)
    prods_x = torch.mm(norms_x, norms_x.t())
    dists_x = p2_norm_x + p2_norm_x.t() - 2 * prods_x

    p2_norm_y = Y.pow(2).sum(1).unsqueeze(0)
    norms_y = X.sum(1).unsqueeze(0)
    prods_y = torch.mm(norms_y, norms_y.t())
    dists_y = p2_norm_y + p2_norm_y.t() - 2 * prods_y

    dot_prd = torch.mm(norms_x, norms_y.t())
    dists_c = p2_norm_x + p2_norm_y.t() - 2 * dot_prd

    stats = 0
    for scale in [.1, .2, .5, 1., 2., 5., 10.]:
        C = 2 * h_dim * 1.0 * scale
        res1 = C / (C + dists_x)
        res1 += C / (C + dists_y)

        if torch.cuda.is_available():
            res1 = (1 - torch.eye(batch_size).cuda()) * res1
        else:
            res1 = (1 - torch.eye(batch_size)) * res1

        res1 = res1.sum() / (batch_size - 1)
        res2 = C / (C + dists_c)
        res2 = res2.sum() * 2. / (batch_size)
        stats += res1 - res2
    return stats


# Maximum Mean Discrepancy between z and a reference distribution
# This term goes to zero if z is perfectly normal (with variance sigma**2)
def mmd_normal_penalty(z, sigma=1.0):
    batch_size, latent_dim = z.shape
    z_fake = torch.randn(batch_size, latent_dim).cuda() * sigma
    #z_fake = norm(z_fake)
    mmd_loss = -imq_kernel(z, z_fake, h_dim=latent_dim)
    return mmd_loss.mean()


# Normalize a batch of latent points to the unit hypersphere
def norm(x):
    norm = torch.norm(x, p=2, dim=1)
    x = x / (norm.expand(1, -1).t() + .0001)
    return x


def main():
    batch_size = 64
    latent_dim = 12
    true_latent_dim = 4
    num_actions = 4
    train_iters = 100 * 1000
    encoder = Encoder(latent_dim)
    decoder = Decoder(latent_dim)
    transition = Transition(latent_dim, num_actions)
    blur = GaussianSmoothing(channels=3, kernel_size=11, sigma=4.)
    higgins_scores = []

    #load_from_dir = '/mnt/nfs/experiments/demo_2018_12_12/scm-gan_81bd12cd'
    load_from_dir = '.'
    if load_from_dir is not None and 'model-encoder.pth' in os.listdir(load_from_dir):
        print('Loading models from directory {}'.format(load_from_dir))
        encoder.load_state_dict(torch.load(os.path.join(load_from_dir, 'model-encoder.pth')))
        decoder.load_state_dict(torch.load(os.path.join(load_from_dir, 'model-decoder.pth')))
        transition.load_state_dict(torch.load(os.path.join(load_from_dir, 'model-transition.pth')))

    # Train the autoencoder
    opt_enc = torch.optim.Adam(encoder.parameters(), lr=.01)
    opt_dec = torch.optim.Adam(decoder.parameters(), lr=.01)
    opt_trans = torch.optim.Adam(transition.parameters(), lr=.01)
    ts = TimeSeries('Training Model', train_iters)
    for train_iter in range(1, train_iters + 1):
        timesteps = 1 + train_iter // 10000
        encoder.train()
        decoder.train()
        transition.train()
        for model in (encoder, decoder, transition):
            for child in model.children():
                if type(child) == nn.BatchNorm2d or type(child) == nn.BatchNorm1d:
                    child.momentum = 0.1

        opt_enc.zero_grad()
        opt_dec.zero_grad()
        opt_trans.zero_grad()

        states, rewards, dones, actions = datasource.get_trajectories(batch_size, timesteps)
        states = torch.Tensor(states).cuda()
        # states.shape: (batch_size, timesteps, 3, 64, 64)

        # Predict the output of the game
        loss = 0
        z = encoder(states[:, 0])
        ts.collect('encoder z[0] mean', z[0].mean())
        for t in range(timesteps):
            pred_logits, aux_loss = decoder(z, enable_aux_loss=True)
            ts.collect('logits min', pred_logits.min())
            ts.collect('logits max', pred_logits.max())
            ts.collect('aux loss', aux_loss)
            loss += aux_loss

            expected = states[:, t]
            #predicted = torch.sigmoid(pred_logits)
            predicted = pred_logits
            # MSE loss
            #rec_loss = torch.mean((expected - predicted)**2)
            # MSE loss but blurred to prevent pathological behavior
            #rec_loss = torch.mean((blur(expected) - blur(predicted))**2)
            # MSE loss but weighted toward foreground pixels
            error_mask = torch.mean((expected - predicted) ** 2, dim=1)
            foreground_mask = torch.mean(blur(expected), dim=1)
            theta = (train_iter / train_iters)
            error_mask = theta * error_mask + (1 - theta) * (error_mask * foreground_mask)
            rec_loss = torch.mean(error_mask)

            ts.collect('Recon. t={}'.format(t), rec_loss)
            loss += rec_loss

            # Latent regression loss: Don't encode non-visible information
            #z_prime = encoder(decoder(z))
            #latent_regression_loss = torch.mean((z - z_prime)**2)
            #ts.collect('Latent reg. t={}'.format(t), latent_regression_loss)
            #loss += latent_regression_loss

            # Predict the next latent point
            onehot_a = torch.eye(num_actions)[actions[:, t]].cuda()
            z = transition(z, onehot_a)

            # Maximum Mean Discrepancy: Regularization toward gaussian
            # mmd_loss = mmd_normal_penalty(z)
            # ts.collect('MMD Loss t={}'.format(t), mmd_loss)
            # loss += mmd_loss

        loss.backward()

        ts.collect('rgb weight std.', decoder.to_rgb.weight.std())

        opt_enc.step()
        opt_dec.step()
        opt_trans.step()
        ts.print_every(2)

        encoder.eval()
        decoder.eval()
        transition.eval()
        for model in (encoder, decoder, transition):
            for child in model.children():
                if type(child) == nn.BatchNorm2d or type(child) == nn.BatchNorm1d:
                    child.momentum = 0

        if train_iter % 100 == 0:
            vis = ((expected - predicted)**2)[:1]
            imutil.show(vis, filename='reconstruction_error.png')

        if train_iter % 100 == 0:
            visualize_reconstruction(encoder, decoder, states, train_iter=train_iter)

        # Periodically generate latent space traversals
        if train_iter % 1000 == 0:
            visualize_latent_space(states, encoder, decoder, latent_dim=latent_dim, train_iter=train_iter)

        # Periodically save the network
        if train_iter % 2000 == 0:
            print('Saving networks to filesystem...')
            torch.save(transition.state_dict(), 'model-transition.pth')
            torch.save(encoder.state_dict(), 'model-encoder.pth')
            torch.save(decoder.state_dict(), 'model-decoder.pth')

        # Periodically generate simulations of the future
        if train_iter % 2000 == 0:
            visualize_forward_simulation(datasource, encoder, decoder, transition, train_iter)

        # Periodically compute the Higgins score
        if train_iter % 10000 == 0:
            if not hasattr(datasource, 'simulator'):
                print('Datasource {} does not support direct simulation, skipping disentanglement metrics'.format(datasource.__name__))
            else:
                trained_score = higgins_metric(datasource.simulator, true_latent_dim, encoder, latent_dim)
                higgins_scores.append(trained_score)
                print('Higgins metric before training: {}'.format(higgins_scores[0]))
                print('Higgins metric after training {} iters: {}'.format(train_iter, higgins_scores[-1]))
                print('Best Higgins: {}'.format(max(higgins_scores)))
                ts.collect('Higgins Metric', trained_score)
    print(ts)
    print('Finished')


def visualize_reconstruction(encoder, decoder, states, train_iter=0):
    # Image of reconstruction
    filename = 'vis_iter_{:06d}.png'.format(train_iter)
    ground_truth = states[:, 0]
    tag = 'iter_{:06d}'.format(train_iter // 1000 * 1000)
    logits = decoder(encoder(ground_truth, visual_tag=tag), visual_tag=tag)
    #reconstructed = torch.sigmoid(logits)
    reconstructed = logits
    img = torch.cat((ground_truth[:4], reconstructed[:4]), dim=3)
    caption = 'D(E(x)) iter {}'.format(train_iter)
    imutil.show(img, resize_to=(640, 360), img_padding=4,
                filename='visual_reconstruction_{}.png'.format(tag),
                caption=caption, font_size=10)


def visualize_latent_space(states, encoder, decoder, latent_dim, train_iter=0, frames=120, img_size=800):
    # Create a "batch" containing copies of the same image, one per latent dimension
    ground_truth = states[:, 0]

    # Hack
    latent_dim //= 2

    for i in range(1, latent_dim):
        ground_truth[i] = ground_truth[0]
    zt = encoder(ground_truth)
    minval, maxval = decoder.to_categorical.rho.min(), decoder.to_categorical.rho.max()

    # Generate L videos, one per latent dimension
    vid = imutil.Video('latent_traversal_dims_{:04d}_iter_{:06d}'.format(latent_dim, train_iter))
    for frame_idx in range(frames):
        for z_idx in range(latent_dim):
            z_val = (frame_idx / frames) * (maxval - minval) + minval
            zt[z_idx, z_idx] = z_val
        #output = torch.sigmoid(decoder(zt))
        output = decoder(zt)[:latent_dim]
        #reconstructed = torch.sigmoid(decoder(encoder(ground_truth)))
        reconstructed = decoder(encoder(ground_truth))
        video_frame = torch.cat([ground_truth[:1], reconstructed[:1], output], dim=0)
        caption = '{}/{} z range [{:.02f} {:.02f}]'.format(frame_idx, frames, minval, maxval)
        vid.write_frame(video_frame, resize_to=(img_size,img_size), caption=caption, img_padding=8)
    vid.finish()


def visualize_forward_simulation(datasource, encoder, decoder, transition, train_iter=0, timesteps=60, num_actions=4):
    start_time = time.time()
    print('Starting trajectory simulation for {} frames'.format(timesteps))
    states, rewards, dones, actions = datasource.get_trajectories(batch_size=64, timesteps=timesteps)
    states = torch.Tensor(states).cuda()
    vid = imutil.Video('simulation_iter_{:06d}.mp4'.format(train_iter), framerate=3)
    z = encoder(states[:, 0])
    for t in range(timesteps):
        #x_t = torch.sigmoid(decoder(z))
        x_t = decoder(z)
        img = torch.cat((states[:, t][:4], x_t[:4]), dim=3)
        caption = 'Pred. t+{} a={}'.format(t, actions[:4, t])
        vid.write_frame(img, caption=caption, img_padding=8, font_size=10, resize_to=(800,400))
        # Predict the next latent point
        onehot_a = torch.eye(num_actions)[actions[:, t]].cuda()
        z = transition(z, onehot_a)
    vid.finish()
    print('Finished trajectory simulation in {:.02f}s'.format(time.time() - start_time))


if __name__ == '__main__':
    main()
