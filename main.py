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

from higgins import higgins_metric

from importlib import import_module
datasource = import_module('envs.' + sys.argv[1])


class Transition(nn.Module):
    def __init__(self, latent_size, num_actions, k=64):
        super().__init__()
        # Input: State + Action
        # Output: State
        self.latent_size = latent_size
        self.k = k
        self.input_dim = latent_size + num_actions
        self.to_categorical = RealToCategorical(latent_size, k)
        self.fc1 = nn.Linear(num_actions + self.latent_size*k, 256)
        self.fc2 = nn.Linear(256, latent_size * k)
        self.to_dense = nn.Linear(latent_size*k, latent_size)
        self.cuda()

    def forward(self, z, actions, evaluation=False):
        expanded = self.to_categorical(z)
        expanded = expanded.view(-1, self.latent_size*self.k)
        x = torch.cat([expanded, actions], dim=1)
        x = self.fc1(x)
        x = F.leaky_relu(x, 0.2)
        x = self.fc2(x)
        x = x.view(len(z), self.latent_size, self.k)
        x = torch.softmax(x, dim=2)
        x = x.view(len(z), self.latent_size*self.k)
        x = self.to_dense(x)
        x = norm(x)
        return x


class Encoder(nn.Module):
    def __init__(self, latent_size, k=64):
        super().__init__()
        self.latent_size = latent_size
        self.k = k
        # Bx1x64x64
        self.conv1 = nn.Conv2d(3, 8, 4, stride=2, padding=1)
        self.bn_conv1 = nn.BatchNorm2d(8)
        # Bx8x32x32
        self.conv2 = nn.Conv2d(8, 32, 4, stride=2, padding=1)
        self.bn_conv2 = nn.BatchNorm2d(32)

        self.fc1 = nn.Linear(32*16*16, 196)
        self.bn1 = nn.BatchNorm1d(196)
        self.fc2 = nn.Linear(196, k*latent_size)
        self.to_dense = CategoricalToReal(self.latent_size, self.k)

        # Bxlatent_size
        self.cuda()

    def forward(self, x, visual_tag=None):
        # Input: B x 1 x 64 x 64
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
        x = x.view(-1, self.latent_size, self.k)
        if visual_tag:
            imutil.show(x[0], resize_to=(self.k*10, self.latent_size*10),
                        filename="visual_{}.png".format(visual_tag),
                        caption="Encoder latent space")
        x = F.softmax(x, dim=2)
        x = self.to_dense(x)
        return x


class Decoder(nn.Module):
    def __init__(self, latent_size, k=64, m=3):
        super().__init__()
        self.latent_size = latent_size
        num_places = latent_size // 2
        # K is the number of buckets to use when discretizing
        self.k = k
        # M is the number of kinds of things that there can be
        self.m = m

        self.to_categorical = RealToCategorical(z=num_places, k=k)
        self.bn_places = nn.BatchNorm2d(latent_size)

        self.things_fc1 = nn.Linear(num_places, 128)
        self.things_bn1 = nn.BatchNorm1d(128)
        self.things_fc2 = nn.Linear(128, num_places * m)

        self.conv1 = nn.Conv2d(self.m, 32, 4, stride=2, padding=1)
        self.bn_conv1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 4, stride=2, padding=1)
        self.bn_conv2 = nn.BatchNorm2d(32)
        self.conv3 = nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1)
        self.bn_conv3 = nn.BatchNorm2d(32)
        self.conv4 = nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1)
        self.cuda()

    def forward(self, z, visual_tag=None):
        # The world consists of things in places.
        num_places = self.latent_size // 2
        z_places, z_things = z[:, :num_places], z[:, num_places:]

        # Compute places as ring of outer products, one place per latent dim
        x_cat = self.to_categorical(z_places)
        if visual_tag:
            imutil.show(x_cat[0], resize_to=(self.k*10, self.latent_size*10),
                        caption="Latent Code",
                        filename='visual_to_cat_{}.png'.format(visual_tag))
        places = torch.zeros((len(x_cat), num_places, self.k, self.k)).cuda()
        for i in range(0, num_places):
            shifted = [x_cat[:,i], x_cat[:,i-1]]
            places[:, i]   = torch.einsum('ij,ik->ijk', shifted)
        places = places - places.min()
        places = places / places.max()
        if visual_tag:
            cap = 'Decoder normalized places'
            imutil.show(places[0], filename='visual_places_{}.png'.format(visual_tag), resize_to=(256,256), caption=cap)

        # There are M things, and each thing is in exactly one place at any given time
        things = self.things_fc1(z_things)
        things = self.things_bn1(things)
        things = F.leaky_relu(things, 0.2)
        things = self.things_fc2(things)
        things = things.view(-1, num_places, self.m)
        # (batch, number_of_things, number_of_places)
        things = F.softmax(things, dim=2)

        # Draw things in places
        x = torch.einsum('bpwh,bpm->bmwh', [places, things])
        if visual_tag:
            cap = 'Things+Places Map'
            imutil.show(x[0], filename='visual_conv0_{}.png'.format(visual_tag), resize_to=(256,256), caption=cap)

        x = self.conv1(x)
        x = self.bn_conv1(x)
        x = F.leaky_relu(x, 0.2)
        if visual_tag:
            cap = 'Decoder conv1'
            imutil.show(x[0], filename='visual_conv1_{}.png'.format(visual_tag), resize_to=(256,256), caption=cap)

        #x = self.conv2(x)
        #x = self.bn_conv2(x)
        #x = F.leaky_relu(x, 0.2)
        #x = self.conv3(x)
        #x = F.leaky_relu(x, 0.2)
        x = self.conv4(x)
        if visual_tag:
            cap = 'Decoder conv4'
            imutil.show(x[0], filename='visual_conv4_{}.png'.format(visual_tag), resize_to=(256,256), caption=cap)
        return x


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
        rho = torch.arange(-1, 1, 2/k).unsqueeze(0).repeat(z, 1).cuda()
        self.particles = torch.nn.Parameter(rho)
        # For fixed particle positions
        #self.particles = rho

        # Sharpness/scale parameter
        eta = torch.ones(self.z).cuda() * 30
        self.eta = torch.nn.Parameter(eta)
        self.cuda()

    def forward(self, x, thermometer=False):
        # x is a real-valued tensor size (batch, Z)
        batch_size = len(x)
        # Broadcast x to (batch, Z, K)
        perceived_locations = x.unsqueeze(-1).repeat(1, 1, self.k)
        reference_locations = self.particles.unsqueeze(0).repeat(batch_size, 1, 1)
        distances = (perceived_locations - reference_locations) ** 2
        # IMQ kernel
        if self.kernel == 'inverse_multiquadratic':
            scale = (1 / eta)
            kern = scale / (scale + distances)
        elif self.kernel == 'gaussian':
            dist_kern = torch.einsum('blk,l->blk', [distances, self.eta])
            kern = torch.exp(-dist_kern)
        # Output is a category between 1 and K, for each of the Z real values
        probs = torch.softmax(kern, dim=2)
        return probs


class CategoricalToReal(nn.Module):
    def __init__(self, z, k, kernel='gaussian'):
        super().__init__()
        self.z = z
        self.k = k
        rho = torch.arange(-1, 1, 2/k).unsqueeze(0).repeat(z, 1).cuda()
        self.particles = torch.nn.Parameter(rho)
        # For fixed particle positions
        #self.particles = rho
        self.cuda()

    def forward(self, x):
        # x is shape (batch, self.k)
        batch_size, z, k = x.shape
        particle_weights = self.particles.unsqueeze(0).repeat(batch_size, 1, 1)
        return torch.einsum('blk,blk->bl', [x, particle_weights])


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
    z_fake = norm(z_fake)
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
    timesteps = 4
    num_actions = 4
    train_iters = 20 * 1000
    encoder = Encoder(latent_dim)
    decoder = Decoder(latent_dim)
    transition = Transition(latent_dim, num_actions)
    blur = GaussianSmoothing(channels=3, kernel_size=11, sigma=4.)
    higgins_scores = []

    #load_from_dir = '/mnt/nfs/experiments/default/scm-gan_06a94339'
    load_from_dir = '.'
    if load_from_dir is not None and 'model-encoder.pth' in os.listdir(load_from_dir):
        print('Loading models from directory {}'.format(load_from_dir))
        encoder.load_state_dict(torch.load(os.path.join(load_from_dir, 'model-encoder.pth')))
        decoder.load_state_dict(torch.load(os.path.join(load_from_dir, 'model-decoder.pth')))
        transition.load_state_dict(torch.load(os.path.join(load_from_dir, 'model-transition.pth')))

    # Train the autoencoder
    opt_enc = torch.optim.Adam(encoder.parameters(), lr=.001)
    opt_dec = torch.optim.Adam(decoder.parameters(), lr=.001)
    opt_trans = torch.optim.Adam(transition.parameters(), lr=.001)
    ts = TimeSeries('Training Model', train_iters)
    for train_iter in range(1, train_iters + 1):
        encoder.train()
        decoder.train()
        transition.train()

        opt_enc.zero_grad()
        opt_dec.zero_grad()
        opt_trans.zero_grad()

        states, rewards, dones, actions = datasource.get_trajectories(batch_size, timesteps)
        states = torch.Tensor(states).cuda()
        # states.shape: (batch_size, timesteps, 3, 64, 64)

        # Predict the output of the game
        loss = 0
        z = encoder(states[:, 0])
        for t in range(timesteps):
            pred_logits = decoder(z)

            expected = states[:, t]
            predicted = torch.sigmoid(pred_logits)
            # MSE loss
            #rec_loss = torch.mean((expected - predicted)**2)
            # MSE loss but blurred to prevent pathological behavior
            #rec_loss = torch.mean((blur(expected) - blur(predicted))**2)
            # MSE loss but weighted toward foreground pixels
            error_mask = torch.mean((expected - predicted) ** 2, dim=1)
            foreground_mask = torch.sqrt(torch.mean(blur(expected), dim=1))
            error_mask = 0.01 * error_mask + 0.99 * (error_mask * foreground_mask)
            rec_loss = torch.mean(error_mask)

            ts.collect('Recon. t={}'.format(t), rec_loss)
            loss += rec_loss
            # Predict the next latent point
            onehot_a = torch.eye(num_actions)[actions[:, t]].cuda()
            z = transition(z, onehot_a)
            # mmd_loss = mmd_normal_penalty(z)
            # ts.collect('MMD Loss t={}'.format(t), mmd_loss)
            # loss += mmd_loss
        loss.backward()

        opt_enc.step()
        opt_dec.step()
        opt_trans.step()
        ts.print_every(2)

        encoder.eval()
        decoder.eval()
        transition.eval()

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
    reconstructed = torch.sigmoid(logits)
    img = torch.cat((ground_truth[:4], reconstructed[:4]), dim=3)
    caption = 'D(E(x)) iter {}'.format(train_iter)
    imutil.show(img, resize_to=(640, 360), img_padding=4,
                filename='visual_reconstruction_{}.png'.format(tag),
                caption=caption, font_size=10)


def visualize_latent_space(states, encoder, decoder, latent_dim, train_iter=0, frames=120, img_size=800):
    # Create a "batch" containing copies of the same image, one per latent dimension
    ground_truth = states[:, 0]
    for i in range(1, latent_dim):
        ground_truth[i] = ground_truth[0]
    zt = encoder(ground_truth[:latent_dim])
    minval, maxval = decoder.to_categorical.particles.min(), decoder.to_categorical.particles.max()

    # Generate L videos, one per latent dimension
    vid = imutil.Video('latent_traversal_dims_{:04d}_iter_{:06d}'.format(latent_dim, train_iter))
    for frame_idx in range(frames):
        for z_idx in range(latent_dim):
            z_val = (frame_idx / frames) * (maxval - minval) + minval
            zt[z_idx, z_idx] = z_val
        output = torch.sigmoid(decoder(zt))
        caption = '{}/{} z range [{:.02f} {:.02f}]'.format(frame_idx, frames, minval, maxval)
        vid.write_frame(output, resize_to=(img_size,img_size), caption=caption, img_padding=8)
    vid.finish()


def visualize_forward_simulation(datasource, encoder, decoder, transition, train_iter=0, timesteps=60, num_actions=4):
    start_time = time.time()
    print('Starting trajectory simulation for {} frames'.format(timesteps))
    states, rewards, dones, actions = datasource.get_trajectories(batch_size=4, timesteps=timesteps)
    states = torch.Tensor(states).cuda()
    vid = imutil.Video('simulation_iter_{:06d}.mp4'.format(train_iter), framerate=3)
    z = encoder(states[:, 0])
    for t in range(timesteps):
        x_t = torch.sigmoid(decoder(z))
        img = torch.cat((states[:, t], x_t), dim=3)
        caption = 'Pred. t+{} a={}'.format(t, actions[:, t])
        vid.write_frame(img, caption=caption, img_padding=8, font_size=10, resize_to=(800,400))
        # Predict the next latent point
        onehot_a = torch.eye(num_actions)[actions[:, t]].cuda()
        z = transition(z, onehot_a, evaluation=True)
    vid.finish()
    print('Finished trajectory simulation in {:.02f}s'.format(time.time() - start_time))


if __name__ == '__main__':
    main()
