import argparse
from importlib import import_module

import time
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import sc2env
import imutil
from logutil import TimeSeries

import models
from causal_graph import render_causal_graph
from higgins import higgins_metric_conv
from utils import cov
from tqdm import tqdm


parser = argparse.ArgumentParser(description="Learn to model a sequential environment")
parser.add_argument('--env', required=True, help='One of: boxes, minipong, Pong-v0, etc (see envs/ for list)')
parser.add_argument('--load-from', required=True, help='Directory containing .pth models (default: .)')
args = parser.parse_args()


def select_environment(env_name):
    if env_name.endswith('v0') or env_name.endswith('v4'):
        datasource = import_module('envs.gym_make')
        datasource.ENV_NAME = env_name
    else:
        datasource = import_module('envs.' + env_name)
    return datasource


def main():
    batch_size = 32
    latent_dim = 16
    train_iters = 100 * 1000

    datasource = select_environment(args.env)
    num_actions = datasource.NUM_ACTIONS
    encoder = models.Encoder(latent_dim)
    decoder = models.Decoder(latent_dim)
    reward_predictor = models.RewardPredictor(latent_dim)
    discriminator = models.Discriminator()
    rgb_decoder = models.RGBDecoder()
    transition = models.Transition(latent_dim, num_actions)

    #load_from_dir = '.'
    #load_from_dir = '/mnt/nfs/experiments/default/scm-gan_07dc24bf'
    load_from_dir = args.load_from or '.'
    if load_from_dir is not None and 'model-encoder.pth' in os.listdir(load_from_dir):
        print('Loading models from directory {}'.format(load_from_dir))
        encoder.load_state_dict(torch.load(os.path.join(load_from_dir, 'model-encoder.pth')))
        decoder.load_state_dict(torch.load(os.path.join(load_from_dir, 'model-decoder.pth')))
        transition.load_state_dict(torch.load(os.path.join(load_from_dir, 'model-transition.pth')))
        discriminator.load_state_dict(torch.load(os.path.join(load_from_dir, 'model-discriminator.pth')))
        reward_predictor.load_state_dict(torch.load(os.path.join(load_from_dir, 'model-reward_predictor.pth')))
        rgb_decoder.load_state_dict(torch.load(os.path.join(load_from_dir, 'model-rgb_decoder.pth')))

    # Train the autoencoder
    opt_enc = torch.optim.Adam(encoder.parameters(), lr=.001)
    opt_dec = torch.optim.Adam(decoder.parameters(), lr=.001)
    opt_trans = torch.optim.Adam(transition.parameters(), lr=.001)
    opt_disc = torch.optim.Adam(discriminator.parameters(), lr=.001)
    opt_pred = torch.optim.Adam(reward_predictor.parameters(), lr=.001)
    opt_rgb = torch.optim.Adam(rgb_decoder.parameters(), lr=.01)
    ts = TimeSeries('Training Model', train_iters, tensorboard=True)

    # Blur for foreground mask
    blur = models.GaussianSmoothing(channels=1, kernel_size=5, sigma=3)

    for train_iter in range(train_iters):
        theta = (train_iter / train_iters)
        prediction_horizon = 5 + int(5 * theta)

        train_mode([encoder, decoder, rgb_decoder, transition, discriminator])

        # Train encoder/transition/decoder
        opt_enc.zero_grad()
        opt_dec.zero_grad()
        opt_trans.zero_grad()
        opt_pred.zero_grad()

        states, rgb_states, rewards, dones, actions = datasource.get_trajectories(batch_size, prediction_horizon)
        states = torch.Tensor(states).cuda()
        rgb_states = torch.Tensor(rgb_states.transpose(0, 1, 4, 2, 3)).cuda()
        rewards = torch.Tensor(rewards).cuda()
        dones = torch.Tensor(dones.astype(int)).cuda()

        # Encode the initial state (using the first 3 frames)
        z = encoder(states[:, 0:3])
        # Given t, t+1, t+2, encoder outputs the state at time t+1
        # We then step forward one timestep (from t+1) to predict t+2
        # actions[:, 2] is the action taken after seeing s_2- ie. a_t = \pi(s_t)
        onehot_a = torch.eye(num_actions)[actions[:, 1]].cuda()
        z = transition(z, onehot_a)
        z0 = z.clone()

        # Keep track of "done" states to stop predicting a trajectory
        #  once it reaches the end of the game
        active_mask = torch.ones(batch_size).cuda()

        loss = 0
        # Given the state encoded at t=2, predict state at t=3, t=4, ...
        for t in range(2, prediction_horizon):
            active_mask = active_mask * (1 - dones[:, t])

            # Predict reward
            expected_reward = reward_predictor(z)
            actual_reward = rewards[:, t]
            reward_difference = torch.mean(torch.abs(expected_reward - actual_reward) * active_mask)
            ts.collect('Rd Loss t={}'.format(t), reward_difference)
            loss += .0 * reward_difference

            # Reconstruction loss
            expected = states[:, t]
            predicted = decoder(z)
            #bg_mse_multiplier = 0.1
            #foreground_mask = (1 - bg_mse_multiplier) * blur(expected.mean(dim=-3, keepdim=True)**2) + bg_mse_multiplier
            #mse_difference = (foreground_mask * (expected - predicted)**2).mean(dim=-1).mean(dim=-1).mean(dim=-1)
            #rec_loss = torch.mean(mse_difference * active_mask)
            rec_loss = torch.mean((expected - predicted) **2)
            ts.collect('MSE t={}'.format(t), rec_loss)
            loss += rec_loss

            '''
            # Apply activation L1 loss
            l1_values = z.abs().mean(-1).mean(-1).mean(-1)
            l1_loss = torch.mean(l1_values * active_mask)
            ts.collect('L1 t={}'.format(t), l1_loss)
            loss += .01 * theta * l1_loss
            '''

            # Spatially-Coherent Log-Determinant independence loss
            # Sample 1000 random latent vector spatial points from the batch
            #latent_vectors = blur(z).permute(0, 2, 3, 1).contiguous().view(-1, latent_dim)
            #rand_indices = np.random.randint(0, len(latent_vectors), size=(1000,))
            #z_samples = latent_vectors[rand_indices]
            #covariance = cov(z_samples)
            #eps = 1e-5
            #log_det = -torch.log(torch.det(covariance / covariance.max()) + eps)
            #log_det_penalty = theta * .01 * log_det
            #ts.collect('Log-Det t={}'.format(t), log_det_penalty)
            #loss += log_det_penalty

            # Predict transition
            onehot_a = torch.eye(num_actions)[actions[:, t]].cuda()
            new_z = transition(z, onehot_a)
            '''
            # Apply transition L1 loss
            t_l1_values = ((new_z - z).abs().mean(-1).mean(-1).mean(-1))
            t_l1_loss = torch.mean(t_l1_values * active_mask)
            ts.collect('T-L1 t={}'.format(t), t_l1_loss)
            loss += .01 * theta * t_l1_loss
            '''
            z = new_z

        '''
        # Apply TD-RNN loss
        # Predict state t+3 by T(E(s_2), a_2) and also by T(T(E(s_1),a_1),a_2)
        # They should be consistent: long-term expectations should match later short-term expectations
        twostep_prediction = transition(z0, torch.eye(num_actions)[actions[:, 2]].cuda())
        twostep_prediction = transition(twostep_prediction, torch.eye(num_actions)[actions[:, 3]].cuda())
        onestep_prediction = encoder(states[:, 1:4])
        onestep_prediction = transition(encoder(states[:, 1:4]), torch.eye(num_actions)[actions[:, 2]].cuda())
        consistency_loss = torch.mean((twostep_prediction - onestep_prediction)**2)
        ts.collect('TDC 2:1', consistency_loss)
        loss += .1 * theta * consistency_loss
        '''

        loss.backward()

        opt_enc.step()
        opt_dec.step()
        opt_trans.step()
        opt_pred.step()
        ts.print_every(2)

        ##### Separately, train an RGB-decoder
        ##### This converts from features to pixels
        opt_rgb.zero_grad()
        expected = rgb_states[:, 2]
        actual = rgb_decoder(decoder(z0).detach())
        rgb_loss = torch.mean((expected - actual)**2)
        rgb_loss.backward()
        opt_rgb.step()

        if train_iter % 100 == 0:
            print('Evaluating networks...')
            test_mode([encoder, decoder, rgb_decoder, transition, discriminator])
            # TODO: visualize pysc2 features
            filename = 'rgb_reconstruction_iter_{:06d}.png'.format(train_iter)
            caption = 'Left: True, Right: Simulated'
            pixels = torch.cat([expected[0], actual[0]], dim=-1)
            imutil.show(pixels * 255., filename=filename, caption=caption, normalize=False)

            """
            # Periodically generate visualizations
            compute_causal_graph(encoder, transition, datasource, iter=train_iter)
            visualize_reconstruction(datasource, encoder, decoder, transition, train_iter=train_iter)
            visualize_forward_simulation(datasource, encoder, decoder, transition, reward_predictor, train_iter, num_actions=num_actions)

            # Periodically compute expensive metrics
            if hasattr(datasource, 'simulator'):
                disentanglement_score = higgins_metric_conv(datasource.simulator, datasource.TRUE_LATENT_DIM, encoder, latent_dim)
            """
            print('Saving networks to filesystem...')
            torch.save(transition.state_dict(), 'model-transition.pth')
            torch.save(encoder.state_dict(), 'model-encoder.pth')
            torch.save(decoder.state_dict(), 'model-decoder.pth')
            torch.save(discriminator.state_dict(), 'model-discriminator.pth')
            torch.save(reward_predictor.state_dict(), 'model-reward_predictor.pth')
            torch.save(rgb_decoder.state_dict(), 'model-rgb_decoder.pth')


    print(ts)
    print('Finished')


def test_mode(networks):
    for net in networks:
        net.eval()
        for child in net.children():
            if type(child) == nn.BatchNorm2d or type(child) == nn.BatchNorm1d:
                child.momentum = 0


def train_mode(networks):
    for net in networks:
        net.train()
        for child in net.children():
            if type(child) == nn.BatchNorm2d or type(child) == nn.BatchNorm1d:
                child.momentum = 0.1


def compute_causal_graph(encoder, transition, datasource, iter=0):
    # Max over 10 runs
    weights_runs = []
    for i in range(10):
        src_z, onehot_a = sample_transition(encoder, transition, datasource)
        causal_edge_weights = compute_causal_edge_weights(src_z, transition, onehot_a)
        weights_runs.append(causal_edge_weights)
    causal_edge_weights = np.max(weights_runs, axis=0)
    imutil.show(causal_edge_weights, resize_to=(256,256),
                filename='causal_matrix_iter_{:06d}.png'.format(iter))

    latent_dim = src_z.shape[1]
    print('Causal Graph Edge Weights')
    print('Latent Factor -> Latent Factor dim={}'.format(latent_dim))
    for i in range(causal_edge_weights.shape[0]):
        for j in range(causal_edge_weights.shape[1]):
            print('{:.03f}\t'.format(causal_edge_weights[i,j]), end='')
        print('')
    graph_img = render_causal_graph(causal_edge_weights)
    imutil.show(graph_img, filename='causal_graph_iter_{:06d}.png'.format(iter))


def sample_transition(encoder, transition, datasource, batch_size=32):
    horizon = 5  # 3 frame encoder input followed by two predicted steps
    num_actions = datasource.NUM_ACTIONS
    states, rewards, dones, actions = datasource.get_trajectories(batch_size, horizon)
    states = torch.Tensor(states).cuda()
    rewards = torch.Tensor(rewards).cuda()
    dones = torch.Tensor(dones.astype(int)).cuda()

    # Start with latent point t=3
    z = encoder(states[:, 0:3])
    z = transition(z, torch.eye(num_actions)[actions[:,2]].cuda())
    latent_dim = z.shape[1]

    # Now discard t=3 because the agent gets ground truth for it
    # Compare z at t=4 and t=5, the first two predicted timesteps
    src_z = transition(z, torch.eye(num_actions)[actions[:, 3]].cuda())
    onehot_a = torch.eye(num_actions)[actions[:, 4]].cuda()
    return src_z, onehot_a


def compute_causal_edge_weights(src_z, transition, onehot_a):
    latent_dim = src_z.shape[1]
    dst_z = transition(src_z, onehot_a)

    # Edge weights for the causal graph: close-to-zero weights can be pruned
    causal_edge_weights = np.zeros(shape=(latent_dim, latent_dim))

    # For each latent factor, check which other factors it "causes"
    # by computing a counterfactual s_{t+1}
    #print("Generating counterfactual perturbations for latent factors dim {}".format(latent_dim))
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
            #print("Factor {} -> Factor {} causal strength: {:.04f}".format(src_factor_idx, dst_factor_idx, edge_weight))
            causal_edge_weights[src_factor_idx, dst_factor_idx] = edge_weight
    #print("Finished generating counterfactual perturbations")

    #print("Normalizing counterfactual perturbations to max {}".format(causal_edge_weights.max()))
    causal_edge_weights /= causal_edge_weights.max()
    return causal_edge_weights


def visualize_reconstruction(datasource, encoder, decoder, transition, train_iter=0):
    # Image of reconstruction
    filename = 'vis_iter_{:06d}.png'.format(train_iter)
    num_actions = datasource.NUM_ACTIONS
    timesteps = 60
    batch_size = 1
    states, rewards, dones, actions = datasource.get_trajectories(batch_size, timesteps)
    states = torch.Tensor(states).cuda()
    rewards = torch.Tensor(rewards).cuda()
    actions = torch.LongTensor(actions).cuda()
    for offset in [0, 1, 2, 3, 5, 10]:
        print('Generating video for offset {}'.format(offset))
        vid = imutil.Video('prediction_{:02}_iter_{:06d}.mp4'.format(offset, train_iter))
        for t in tqdm(range(3, timesteps - 10)):
            #print('encoding frame {}'.format(t))
            z = encoder(states[:, t-3:t])
            for i in range(offset):
                onehot_a = torch.eye(num_actions)[actions[:, t + i]].cuda()
                #print('\t...predicting frame {}'.format(t + i + 1))
                z = transition(z, onehot_a)
            predicted = decoder(z)
            actual = states[:, t + offset]
            left = imutil.get_pixels(actual[0], normalize=False)
            right = imutil.get_pixels(predicted[0], normalize=False)
            pixels = np.concatenate([left, right], axis=1)
            pixels = np.clip(pixels, 0, 1)
            caption = "Left: True t={} Right: Predicted from t-{}".format(t, offset)
            vid.write_frame(pixels * 255, normalize=False, img_padding=8, resize_to=(800, 400), caption=caption)
        vid.finish()
    print('Finished generating forward-prediction videos')


def visualize_forward_simulation(datasource, encoder, decoder, transition, reward_pred, train_iter=0, timesteps=60, num_actions=4):
    start_time = time.time()
    print('Starting trajectory simulation for {} frames'.format(timesteps))
    states, rewards, dones, actions = datasource.get_trajectories(batch_size=1, timesteps=timesteps)
    states = torch.Tensor(states).cuda()
    vid_simulation = imutil.Video('simulation_only_iter_{:06d}.mp4'.format(train_iter), framerate=3)
    vid_features = imutil.Video('simulation_iter_{:06d}.mp4'.format(train_iter), framerate=3)
    vid_separable_conv = imutil.Video('simulation_separable_iter_{:06d}.mp4'.format(train_iter), framerate=3)
    z = encoder(states[:, :3])
    z = transition(z, torch.eye(num_actions)[actions[:, 2]].cuda())
    z.detach()
    for t in range(3, timesteps - 1):
        x_t, x_t_separable = decoder(z, visualize=True)
        estimated_reward = reward_pred(z)

        # Render top row: real video vs. simulation from initial conditions
        pixel_view = torch.cat((states[:, t], x_t), dim=3)
        caption = 'Pred. t+{} a={} R est={:.2f} R={} min={:.2f} max={:.2f}'.format(
            t, actions[:, t], estimated_reward[0], rewards[:, t], pixel_view.min(), pixel_view.max())
        top_row = imutil.show(pixel_view.clamp_(0,1), caption=caption, img_padding=8, font_size=10,
                              resize_to=(800,400), return_pixels=True, display=False, save=False)
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
