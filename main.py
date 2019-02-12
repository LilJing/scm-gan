import argparse
from importlib import import_module
import sc2env

import time
import math
import os

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

import models

from causal_graph import render_causal_graph



parser = argparse.ArgumentParser(description="Learn to model a sequential environment")
parser.add_argument('--env', required=True, help='One of: boxes, minipong, Pong-v0, etc (see envs/ for list)')
args = parser.parse_args()

def select_environment(env_name):
    if env_name.endswith('v0') or env_name.endswith('v4'):
        datasource = import_module('envs.gym_make')
        datasource.ENV_NAME = env_name
    else:
        datasource = import_module('envs.' + env_name)
    return datasource

datasource = select_environment(args.env)



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


def cov(m, rowvar=False):
    '''Estimate a covariance matrix given data.

    Covariance indicates the level to which two variables vary together.
    If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
    then the covariance matrix element `C_{ij}` is the covariance of
    `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.

    Args:
        m: A 1-D or 2-D array containing multiple variables and observations.
            Each row of `m` represents a variable, and each column a single
            observation of all those variables.
        rowvar: If `rowvar` is True, then each row represents a
            variable, with observations in the columns. Otherwise, the
            relationship is transposed: each column represents a variable,
            while the rows contain observations.

    Returns:
        The covariance matrix of the variables.
    '''
    if m.dim() > 2:
        raise ValueError('m has more than 2 dimensions')
    if m.dim() < 2:
        m = m.view(1, -1)
    if not rowvar and m.size(0) != 1:
        m = m.t()
    # m = m.type(torch.double)  # uncomment this line if desired
    fact = 1.0 / (m.size(1) - 1)
    m -= torch.mean(m, dim=1, keepdim=True)
    mt = m.t()  # if complex: mt = m.t().conj()
    return fact * m.matmul(mt).squeeze()


def main():
    batch_size = 32
    latent_dim = 16
    num_actions = datasource.NUM_ACTIONS
    train_iters = 10 * 1000
    encoder = models.Encoder(latent_dim)
    decoder = models.Decoder(latent_dim)
    discriminator = models.Discriminator()
    transition = models.Transition(latent_dim, num_actions)
    #blur = models.GaussianSmoothing(channels=3, kernel_size=11, sigma=4.)
    higgins_scores = []

    load_from_dir = '.'
    #load_from_dir = '/mnt/nfs/experiments/default/scm-gan_a3ad2d0c'
    if load_from_dir is not None and 'model-encoder.pth' in os.listdir(load_from_dir):
        print('Loading models from directory {}'.format(load_from_dir))
        encoder.load_state_dict(torch.load(os.path.join(load_from_dir, 'model-encoder.pth')))
        decoder.load_state_dict(torch.load(os.path.join(load_from_dir, 'model-decoder.pth')))
        transition.load_state_dict(torch.load(os.path.join(load_from_dir, 'model-transition.pth')))
        discriminator.load_state_dict(torch.load(os.path.join(load_from_dir, 'model-discriminator.pth')))

    # Train the autoencoder
    opt_enc = torch.optim.Adam(encoder.parameters(), lr=.001)
    opt_dec = torch.optim.Adam(decoder.parameters(), lr=.001)
    opt_trans = torch.optim.Adam(transition.parameters(), lr=.001)
    opt_disc = torch.optim.Adam(discriminator.parameters(), lr=.0005)
    ts = TimeSeries('Training Model', train_iters)
    for train_iter in range(1, train_iters):
        #theta = (train_iter / train_iters)
        theta = 0.5
        timesteps = 5 + int(5 * theta)
        encoder.train()
        decoder.train()
        transition.train()
        discriminator.train()
        for model in (encoder, decoder, transition, discriminator):
            for child in model.children():
                if type(child) == nn.BatchNorm2d or type(child) == nn.BatchNorm1d:
                    child.momentum = 0.1

        """
        # Train discriminator
        states, rewards, dones, actions = datasource.get_trajectories(batch_size, 1)
        states = torch.Tensor(states[:, 0]).cuda()
        opt_disc.zero_grad()
        real_scores = discriminator(states)
        fake_scores = discriminator(decoder(encoder(states)))
        real_loss = torch.mean(F.relu(1 - real_scores))
        fake_loss = torch.mean(F.relu(1 + fake_scores))
        ts.collect('D. real', real_loss)
        ts.collect('D. fake', fake_loss)
        disc_loss = real_loss + fake_loss
        disc_loss.backward()
        opt_disc.step()
        """

        # Train encoder/transition/decoder
        opt_enc.zero_grad()
        opt_dec.zero_grad()
        opt_trans.zero_grad()

        """
        states, rewards, dones, actions = datasource.get_trajectories(batch_size, 1)
        states = torch.Tensor(states[:, 0]).cuda()

        # Train decoder using discriminator
        fake_scores = discriminator(decoder(encoder(states).detach()))
        gen_loss = .0001 * theta * torch.mean(F.relu(1 - fake_scores))
        ts.collect('D. gen', gen_loss)
        gen_loss.backward()
        """

        states, rewards, dones, actions = datasource.get_trajectories(batch_size, timesteps)
        states = torch.Tensor(states).cuda()
        dones = torch.Tensor(dones.astype(int)).cuda()

        # Encode the initial state
        z = encoder(states[:, 0])
        ts.collect('encoder z[0] mean', z[0].mean())

        # Predict forward in time
        loss = 0
        restart_indices = []
        for t in range(timesteps):
            # For episodes that begin at this frame, re-encode z
            restart_indices = dones[:, t].nonzero()[:, 0]
            if len(restart_indices) > 0:
                z = z.clone()
                z[restart_indices] = encoder(states[restart_indices, t])

            predicted = decoder(z)

            # Log-Det independence loss
            # Applies independently to each position in z
            # TODO: sample a subset of z values uniformly from the batch
            #z_samples = z.view(-1, latent_dim).permute(1, 0)[:, :1000].clone()
            #covariance = cov(z_samples)
            # The gradient of -log(det(X_ij)) is just X_ij
            #log_det_penalty = theta * .1 * covariance.mean()
            #ts.collect('Log-Det t={}'.format(t), log_det_penalty)
            #loss += log_det_penalty

            # L1 Sparsity loss
            l1_penalty = theta * .001 * z.abs().mean()
            ts.collect('L1 t={}'.format(t), l1_penalty)
            loss += l1_penalty
            expected = states[:, t]

            # MSE loss
            rec_loss = torch.mean((expected - predicted)**2)

            ts.collect('MSE t={}'.format(t), rec_loss)
            loss += rec_loss

            # Predict the next latent point
            onehot_a = torch.eye(num_actions)[actions[:, t]].cuda()
            new_z = transition(z, onehot_a)

            # Transition L1 sparsity loss
            trans_l1_penalty = theta * .001 * (new_z - z).abs().mean()
            ts.collect('T-L1 t={}'.format(t), trans_l1_penalty)
            loss += trans_l1_penalty
            z = new_z

        loss.backward()

        opt_enc.step()
        opt_dec.step()
        opt_trans.step()
        ts.print_every(2)

        encoder.eval()
        decoder.eval()
        transition.eval()
        discriminator.eval()
        for model in (encoder, decoder, transition, discriminator):
            for child in model.children():
                if type(child) == nn.BatchNorm2d or type(child) == nn.BatchNorm1d:
                    child.momentum = 0

        if train_iter % 100 == 0:
            vis = ((expected - predicted)**2)[:1]
            imutil.show(vis, filename='reconstruction_error.png')

        if train_iter % 100 == 0:
            visualize_reconstruction(encoder, decoder, states, train_iter=train_iter)

        # Periodically save the network
        if train_iter % 100 == 0:
            print('Saving networks to filesystem...')
            torch.save(transition.state_dict(), 'model-transition.pth')
            torch.save(encoder.state_dict(), 'model-encoder.pth')
            torch.save(decoder.state_dict(), 'model-decoder.pth')
            torch.save(discriminator.state_dict(), 'model-discriminator.pth')

        # Periodically generate simulations of the future
        if train_iter % 100 == 0:
            visualize_forward_simulation(datasource, encoder, decoder, transition, train_iter, num_actions=num_actions)

        if train_iter % 1000 == 0:
            compute_causal_graph(encoder, transition, states, actions, latent_dim=latent_dim, num_actions=num_actions, iter=train_iter)

        """
        # Periodically generate latent space traversals
        if train_iter % 1000 == 0:
            visualize_latent_space(states, encoder, decoder, latent_dim=latent_dim, train_iter=train_iter)
        """

    print(ts)
    print('Finished')


def compute_causal_graph(encoder, transition, states, actions, latent_dim, num_actions, iter=0):
    # TODO: manage batch size
    assert len(states) > latent_dim

    # Start with latent point t=0 (note: t=0 is a special case)
    # Note: z_{t=0} is a special case so we use t=1 vs. t=2
    z = encoder(states[:, 0])

    # Compare z at t=1 and t=2 (discard t=0 because it's a special case)
    onehot_a = torch.eye(num_actions)[actions[:, 0]].cuda()
    src_z = transition(z, onehot_a)
    onehot_a = torch.eye(num_actions)[actions[:, 1]].cuda()
    dst_z = transition(src_z, onehot_a)

    # Edge weights for the causal graph: close-to-zero weights can be pruned
    causal_edge_weights = np.zeros(shape=(latent_dim, latent_dim))

    # For each latent factor, check which other factors it "causes"
    # by computing a counterfactual s_{t+1}
    print("Generating counterfactual perturbations for latent factors dim {}".format(latent_dim))
    for src_factor_idx in range(latent_dim):
        # The next timestep (according to our model)
        ground_truth_outcome = dst_z

        # What if z[:,latent_idx] had been erased, set to zero?
        perturbed_src_z = src_z.clone()
        perturbed_src_z[:, src_factor_idx] = 0

        # The counterfactual next timestep (according to our model)
        counterfactual_outcome = transition(perturbed_src_z, onehot_a)

        # Difference between what we normally expect to happen,
        #  and what *would* happen IF NOT FOR the source factor
        cf_difference = (ground_truth_outcome - counterfactual_outcome)**2
        for dst_factor_idx in range(latent_dim):
            edge_weight = float(cf_difference[:,dst_factor_idx].max())
            print("Factor {} -> Factor {} causal strength: {:.04f}".format(
                src_factor_idx, dst_factor_idx, edge_weight))
            causal_edge_weights[src_factor_idx, dst_factor_idx] = edge_weight
    print("Finished generating counterfactual perturbations")

    print("Normalizing counterfactual perturbations to max {}".format(causal_edge_weights.max()))
    causal_edge_weights /= causal_edge_weights.max()

    print('Causal Graph Edge Weights')
    print('Latent Factor -> Latent Factor dim={}'.format(latent_dim))
    for i in range(causal_edge_weights.shape[0]):
        for j in range(causal_edge_weights.shape[1]):
            print('{:.03f}\t'.format(causal_edge_weights[i,j]), end='')
        print('')
    graph_img = render_causal_graph(causal_edge_weights)
    imutil.show(graph_img, filename='causal_graph_iter_{:06d}.png'.format(iter))


def visualize_reconstruction(encoder, decoder, states, train_iter=0):
    # Image of reconstruction
    filename = 'vis_iter_{:06d}.png'.format(train_iter)
    ground_truth = states[:, 0]
    tag = 'iter_{:06d}'.format(train_iter)
    logits = decoder(encoder(ground_truth))
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

    for i in range(1, latent_dim):
        ground_truth[i] = ground_truth[0]
    zt = encoder(ground_truth)
    zt.detach()
    #minval, maxval = decoder.to_categorical.rho.min(), decoder.to_categorical.rho.max()
    minval, maxval = -1, 1

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
        # Clip and scale
        video_frame = 255 * torch.clamp(video_frame, 0, 1)
        vid.write_frame(video_frame, resize_to=(img_size,img_size), caption=caption, img_padding=8, normalize=False)
    vid.finish()


def visualize_forward_simulation(datasource, encoder, decoder, transition, train_iter=0, timesteps=60, num_actions=4):
    start_time = time.time()
    print('Starting trajectory simulation for {} frames'.format(timesteps))
    states, rewards, dones, actions = datasource.get_trajectories(batch_size=1, timesteps=timesteps, random_start=False)
    states = torch.Tensor(states).cuda()
    vid_simulation = imutil.Video('simulation_only_iter_{:06d}.mp4'.format(train_iter), framerate=3)
    vid_features = imutil.Video('simulation_iter_{:06d}.mp4'.format(train_iter), framerate=3)
    vid_separable_conv = imutil.Video('simulation_separable_iter_{:06d}.mp4'.format(train_iter), framerate=3)
    z = encoder(states[:1, 0])
    z.detach()
    for t in range(timesteps - 1):
        x_t, x_t_separable = decoder(z, visualize=True)

        # Render top row: real video vs. simulation from initial conditions
        pixel_view = torch.cat((states[:, t][:1], x_t[:1]), dim=3)
        caption = 'Pred. t+{} a={} min={:.2f} max={:.2f}'.format(t, actions[:1, t], pixel_view.min(), pixel_view.max())
        top_row = imutil.show(pixel_view.clamp_(0,1), caption=caption, img_padding=8, font_size=10, resize_to=(800,400), return_pixels=True, display=False, save=False)
        caption = 'Left: Real          Right: Simulated from initial conditions t={}'.format(t)
        vid_simulation.write_frame(pixel_view.clamp(0, 1), caption=caption, resize_to=(1280,640))

        # Render latent representation of simulation
        bottom_row = imutil.show(z[0], resize_to=(800,800), return_pixels=True, img_padding=8, display=False, save=False)
        vid_features.write_frame(np.concatenate([top_row, bottom_row], axis=0))

        # Render pixels generated from latent representation (groupwise separable)
        separable_output = imutil.show(x_t_separable, resize_to=(800,800), return_pixels=True, img_padding=8, display=False, save=False)
        vid_separable_conv.write_frame(np.concatenate([top_row, separable_output], axis=0))

        # Predict the next latent point
        onehot_a = torch.eye(num_actions)[actions[:, t + 1]].cuda()
        z = transition(z, onehot_a).detach()

        if dones[0, t]:
            break

    vid_simulation.finish()
    vid_features.finish()
    vid_separable_conv.finish()
    print('Finished trajectory simulation in {:.02f}s'.format(time.time() - start_time))


if __name__ == '__main__':
    main()
