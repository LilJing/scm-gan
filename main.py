import time
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
        self.to_categorical = SelfOrganizingBucket(latent_size, k)
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
    def __init__(self, latent_size):
        super().__init__()
        self.latent_size = latent_size
        # Bx1x64x64
        self.conv1 = nn.Conv2d(3, 8, 4, stride=2, padding=1)
        self.bn_conv1 = nn.BatchNorm2d(8)
        # Bx8x32x32
        self.conv2 = nn.Conv2d(8, 32, 4, stride=2, padding=1)
        self.bn_conv2 = nn.BatchNorm2d(32)

        self.fc1 = nn.Linear(32*16*16, 196)
        self.bn1 = nn.BatchNorm1d(196)
        self.fc2 = nn.Linear(196, 196)
        self.bn2 = nn.BatchNorm1d(196)
        self.fc3 = nn.Linear(196, latent_size)

        # Bxlatent_size
        self.cuda()

    def forward(self, x):
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
        x = self.bn2(x)
        x = F.leaky_relu(x)

        x = self.fc3(x)
        x = norm(x)
        return x


class Decoder(nn.Module):
    def __init__(self, latent_size, k=64):
        super().__init__()
        self.latent_size = latent_size
        self.k = k

        self.conv1 = nn.Conv2d(latent_size, 32, 4, stride=2, padding=1)
        self.bn_conv1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 4, stride=2, padding=1)
        self.bn_conv2 = nn.BatchNorm2d(32)
        self.conv3 = nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1)
        self.bn_conv3 = nn.BatchNorm2d(32)
        self.conv4 = nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1)

        self.to_categorical = SelfOrganizingBucket(z=latent_size, k=k)
        self.bn_places = nn.BatchNorm2d(latent_size)
        self.cuda()

    def forward(self, x):
        # The world consists of things in places.
        x = self.to_categorical(x)
        places = torch.zeros((len(x), self.latent_size, self.k, self.k)).cuda()
        # Cycle of outer products
        for i in range(self.latent_size):
            places[:, i] = torch.einsum('ij,ik->ijk', [x[:,i], x[:,i-1]])

        x = F.softmax(places, dim=1)
        x = self.bn_places(x)

        # Given places, make some things
        x = self.conv1(x)
        x = self.bn_conv1(x)
        x = F.leaky_relu(x, 0.2)
        x = self.conv2(x)
        x = self.bn_conv2(x)
        x = F.leaky_relu(x, 0.2)
        x = self.conv3(x)
        x = F.leaky_relu(x, 0.2)
        x = self.conv4(x)
        return x


class SelfOrganizingBucket(nn.Module):
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
        rho = torch.arange(-1, 1, 2/k).unsqueeze(0).repeat(z, 1).cuda()
        self.particles = torch.nn.Parameter(rho)
        #self.particles = rho
        self.cuda()

    def forward(self, x, thermometer=False):
        # x is a real-valued tensor size (batch, Z)
        batch_size = len(x)
        # Broadcast x to (batch, Z, K)
        perceived_locations = x.unsqueeze(-1).repeat(1, 1, self.k)
        reference_locations = self.particles.unsqueeze(0).repeat(batch_size, 1, 1)
        distances = (perceived_locations - reference_locations) ** 2
        # IMQ kernel
        kern = .01 / (.01 + distances)
        # Gaussian RBF kernel
        # kern = torch.exp(-distances)
        # Output is a category between 1 and K, for each of the Z real values
        probs = torch.softmax(kern, dim=2)
        if thermometer:
            therm = probs.clone()
            for i in range(1, self.k):
                therm[:, :, i] = torch.max(probs[:, :, i], therm[:, :, i - 1].clone())
            probs = 1 - therm
        return probs


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
    encoder = Encoder(latent_dim)
    decoder = Decoder(latent_dim)
    transition = Transition(latent_dim, num_actions)
    higgins_scores = []

    #load_from_dir = '/mnt/nfs/experiments/default/scm-gan_06a94339'
    load_from_dir = None
    if load_from_dir is not None:
        encoder.load_state_dict(torch.load(os.path.join(load_from_dir, 'model-encoder.pth')))
        decoder.load_state_dict(torch.load(os.path.join(load_from_dir, 'model-decoder.pth')))
        transition.load_state_dict(torch.load(os.path.join(load_from_dir, 'model-transition.pth')))

    # Train the autoencoder
    opt_enc = torch.optim.Adam(encoder.parameters(), lr=.001)
    opt_dec = torch.optim.Adam(decoder.parameters(), lr=.001)
    opt_trans = torch.optim.Adam(transition.parameters(), lr=.001)
    train_iters = 100 * 1000
    ts = TimeSeries('Training Model', train_iters)
    for train_iter in range(train_iters + 1):
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
            #rec_loss = F.binary_cross_entropy_with_logits(pred_logits, states[:, t])
            rec_loss = torch.mean((states[:, t] - torch.sigmoid(pred_logits))**2)
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

        # Periodically generate latent space traversals
        if train_iter and train_iter % 1000 == 0:
            # Image of reconstruction
            filename = 'vis_iter_{:06d}.jpg'.format(train_iter)
            ground_truth = states[:, 0]
            reconstructed = torch.sigmoid(decoder(encoder(ground_truth)))
            img = torch.cat((ground_truth[:4], reconstructed[:4]), dim=3)
            caption = 'D(E(x)) iter {}'.format(train_iter)
            imutil.show(img, filename=filename, caption=caption, img_padding=4, font_size=10)

            # Video of latent space traversal
            ground_truth = states[:, 0]
            for i in range(1, latent_dim):
                ground_truth[i] = ground_truth[0]
            visualize_latent_space(ground_truth, encoder, decoder, latent_dim=latent_dim, train_iter=train_iter)

        # Periodically save the network
        if train_iter and train_iter % 2000 == 0:
            print('Saving networks to filesystem...')
            torch.save(transition.state_dict(), 'model-transition.pth')
            torch.save(encoder.state_dict(), 'model-encoder.pth')
            torch.save(decoder.state_dict(), 'model-decoder.pth')

        # Periodically generate simulations of the future
        if train_iter and train_iter % 2000 == 0:
            simulate_future(datasource, encoder, decoder, transition, train_iter)

        # Periodically compute the Higgins score
        if train_iter and train_iter % 10000 == 0:
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


def visualize_latent_space(ground_truth, encoder, decoder, latent_dim, train_iter=0, frames=120, img_size=800):
    vid = imutil.Video('latent_traversal_dims_{:04d}_iter_{:06d}'.format(latent_dim, train_iter))
    # Create a "batch" containing copies of the same image, one per latent dimension
    zt = encoder(ground_truth[:latent_dim])
    minval, maxval = decoder.to_categorical.particles.min(), decoder.to_categorical.particles.max()
    for frame_idx in range(frames):
        for z_idx in range(latent_dim):
            z_val = (frame_idx / frames) * (maxval - minval) + minval
            zt[z_idx, z_idx] = z_val
        output = torch.sigmoid(decoder(norm(zt)))
        caption = '{}/{} z range [{:.02f} {:.02f}]'.format(frame_idx, frames, minval, maxval)
        vid.write_frame(output, resize_to=(img_size,img_size), caption=caption, img_padding=8)
    vid.finish()


def simulate_future(datasource, encoder, decoder, transition, train_iter=0, timesteps=60, num_actions=4):
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
