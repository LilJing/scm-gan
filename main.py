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
parser.add_argument('--evaluate', action='store_true', help='If true, evaluate instead of training')
parser.add_argument('--evaluations', type=int, default=1, help='Integer number of evaluations to run')
args = parser.parse_args()


def select_environment(env_name):
    if env_name.endswith('v0') or env_name.endswith('v4'):
        datasource = import_module('envs.gym_make')
        datasource.ENV_NAME = env_name
    else:
        datasource = import_module('envs.' + env_name)
    return datasource


def main():
    latent_dim = 16

    datasource = select_environment(args.env)
    num_actions = datasource.NUM_ACTIONS
    num_rewards = datasource.NUM_REWARDS
    encoder = nn.DataParallel(models.Encoder(latent_dim))
    decoder = nn.DataParallel(models.Decoder(latent_dim))
    reward_predictor = nn.DataParallel(models.RewardPredictor(latent_dim, num_rewards))
    discriminator = nn.DataParallel(models.Discriminator())
    rgb_decoder = nn.DataParallel(models.RGBDecoder())
    transition = nn.DataParallel(models.Transition(latent_dim, num_actions))

    load_from_dir = args.load_from or '.'
    if load_from_dir is not None and 'model-encoder.pth' in os.listdir(load_from_dir):
        print('Loading models from directory {}'.format(load_from_dir))
        encoder.load_state_dict(torch.load(os.path.join(load_from_dir, 'model-encoder.pth')))
        decoder.load_state_dict(torch.load(os.path.join(load_from_dir, 'model-decoder.pth')))
        transition.load_state_dict(torch.load(os.path.join(load_from_dir, 'model-transition.pth')))
        discriminator.load_state_dict(torch.load(os.path.join(load_from_dir, 'model-discriminator.pth')))
        reward_predictor.load_state_dict(torch.load(os.path.join(load_from_dir, 'model-reward_predictor.pth')))
        rgb_decoder.load_state_dict(torch.load(os.path.join(load_from_dir, 'model-rgb_decoder.pth')))

    if args.evaluate:
        for _ in range(args.evaluations):
            play(latent_dim, datasource, num_actions, num_rewards, encoder, decoder,
                 reward_predictor, discriminator, rgb_decoder, transition)
    else:
        train(latent_dim, datasource, num_actions, num_rewards, encoder, decoder,
              reward_predictor, discriminator, rgb_decoder, transition)


def train(latent_dim, datasource, num_actions, num_rewards,
          encoder, decoder, reward_predictor, discriminator, rgb_decoder, transition):
    batch_size = 32
    train_iters = 100 * 1000

    opt_enc = torch.optim.Adam(encoder.parameters(), lr=.001)
    opt_dec = torch.optim.Adam(decoder.parameters(), lr=.001)
    opt_trans = torch.optim.Adam(transition.parameters(), lr=.001)
    opt_disc = torch.optim.Adam(discriminator.parameters(), lr=.001)
    opt_pred = torch.optim.Adam(reward_predictor.parameters(), lr=.001)
    opt_rgb = torch.optim.Adam(rgb_decoder.parameters(), lr=.01)
    ts = TimeSeries('Training Model', train_iters, tensorboard=True)

    for train_iter in range(1, train_iters):
        if train_iter % 1000 == 0:
            print('Evaluating networks...')
            evaluate(datasource, encoder, decoder, rgb_decoder, transition, discriminator, reward_predictor, latent_dim, train_iter=train_iter)
            print('Saving networks to filesystem...')
            torch.save(transition.state_dict(), 'model-transition.pth')
            torch.save(encoder.state_dict(), 'model-encoder.pth')
            torch.save(decoder.state_dict(), 'model-decoder.pth')
            torch.save(discriminator.state_dict(), 'model-discriminator.pth')
            torch.save(reward_predictor.state_dict(), 'model-reward_predictor.pth')
            torch.save(rgb_decoder.state_dict(), 'model-rgb_decoder.pth')

        theta = (train_iter / train_iters)
        prediction_horizon = 5 + int(10 * theta)

        train_mode([encoder, decoder, rgb_decoder, transition, discriminator, reward_predictor])

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
        td_lambda_loss = 0
        lamb = 0.9
        td_z_set = {}
        # Given the state encoded at t=2, predict state at t=3, t=4, ...
        for t in range(2, prediction_horizon):
            active_mask = active_mask * (1 - dones[:, t])

            # Predict reward
            expected_reward = reward_predictor(z)
            actual_reward = rewards[:, t]
            reward_difference = torch.mean(torch.mean((expected_reward - actual_reward)**2, dim=1) * active_mask)
            ts.collect('Rd Loss t={}'.format(t), reward_difference)
            loss += .01 * reward_difference

            # Reconstruction loss
            expected = states[:, t]
            predicted = decoder(z)
            rec_loss = torch.mean((expected - predicted) **2)
            ts.collect('MSE t={}'.format(t), rec_loss)
            loss += rec_loss

            # Apply activation L1 loss
            l1_values = z.abs().mean(-1).mean(-1).mean(-1)
            l1_loss = torch.mean(l1_values * active_mask)
            ts.collect('L1 t={}'.format(t), l1_loss)
            loss += .01 * theta * l1_loss

            # Predict transition to the next state
            onehot_a = torch.eye(num_actions)[actions[:, t]].cuda()
            new_z = transition(z, onehot_a)
            # Apply transition L1 loss
            t_l1_values = ((new_z - z).abs().mean(-1).mean(-1).mean(-1))
            t_l1_loss = torch.mean(t_l1_values * active_mask)
            ts.collect('T-L1 t={}'.format(t), t_l1_loss)
            loss += .01 * theta * t_l1_loss
            z = new_z

            # TD-Lambda Loss
            # State i|j is the predicted state at time i conditioned on observation j
            # Eg. if we predict 2 steps perfectly, then state 3|1 will be equal to state 3|3
            # Also, state 3|1 will be equal to state 3|2
            # Encode the ground-truth state for t_r
            t_r = t
            td_z = encoder(states[:, t_r-2:t_r+1])
            a = torch.eye(num_actions)[actions[:, t_r - 1]].cuda()
            td_z_set[t_r] = transition(td_z, a)

            # For each previous t_left, step forward to t_r
            for t_left in range(2, t_r):
                a = torch.eye(num_actions)[actions[:, t_r - 1]].cuda()
                td_z_set[t_left] = transition(td_z_set[t_left], a)

            # At time t_r, consider each combination (t_a, t_b) where a < b <= r
            # At t_a, we thought t_r would be s_{r|a}
            # But later at t_b, we updated our belief to s_{r|b}
            # Update s_{r|a} to be closer to s_{r|b}, for every b up to and including s_{r|r}
            for t_a in range(2, t_r - 1):
                # for t_b in range(2, t_a):
                # Proper TD: use 4:3, 3:2, 2:1. Not 4:3, 4:2, 4:1, 3:2, 3:1...
                t_b = t_a + 1
                expected = td_z_set[t_a]
                actual = td_z_set[t_b].detach()
                td_loss = torch.mean((expected - actual)**2 * active_mask)

                r_expected = reward_predictor(expected).detach()
                r_actual = reward_predictor(actual)
                r_diffs = torch.mean((r_expected - r_actual)**2, dim=1)
                r_loss = torch.mean(r_diffs * active_mask)
                td_lambda_loss += lamb ** (t_b - 1) * (td_loss + r_loss)

        ts.collect('TD', td_lambda_loss)
        loss += theta * td_lambda_loss
        loss.backward()

        opt_enc.step()
        opt_dec.step()
        opt_trans.step()
        opt_pred.step()

        # Begin StarCraft-Specific Hacks
        # For StarCraftII we use a two-stage decoder
        # The dynamics and planning model operates entirely in pysc2 feature space
        # At the end, we convert features to human-visible RGB
        if train_iter % 4 == 0:
            opt_rgb.zero_grad()
            real_rgb = rgb_states[:, 2]
            simulated_features = decoder(z0).detach()
            simulated_rgb = rgb_decoder(simulated_features)
            rgb_loss = torch.mean((real_rgb - simulated_rgb)**2)
            ts.collect('RGB loss', rgb_loss)
            rgb_loss.backward()
            opt_rgb.step()
        # End StarCraft-Specific Hacks

        ts.print_every(10)
    print(ts)
    print('Finished')


def evaluate(datasource, encoder, decoder, rgb_decoder, transition, discriminator, reward_predictor, latent_dim, train_iter=0):
    print('Evaluating networks...')
    test_mode([encoder, decoder, rgb_decoder, transition, discriminator, reward_predictor])

    # Run visualizations
    #est_reward = format_reward_vector(reward_predictor(z0)[0])
    #caption = 'Left: Input, Right: Generated. iter: {:06d} R: {}'.format(train_iter, est_reward)
    #pixels = torch.cat([expected[0], actual[0]], dim=-1)
    #filename = 'rgb_reconstruction_iter_{:06d}.png'.format(train_iter)
    #imutil.show(pixels * 255., resize_to=(1024, 512), filename=filename, caption=caption, normalize=False)

    #measure_prediction_mse(datasource, encoder, decoder, rgb_decoder, transition, reward_predictor, train_iter, num_factors=latent_dim)
    visualize_forward_simulation(datasource, encoder, decoder.module, rgb_decoder.module, transition, reward_predictor.module, train_iter, num_factors=latent_dim)
    visualize_reconstruction(datasource, encoder, decoder.module, rgb_decoder.module, transition, reward_predictor.module, train_iter=train_iter)

    # Periodically compute expensive metrics
    #if hasattr(datasource, 'simulator'):
    #    disentanglement_score = higgins_metric_conv(datasource.simulator, datasource.TRUE_LATENT_DIM, encoder, latent_dim)


# Apply a simple model-predictive control algorithm using the learned model,
# to take actions that will maximize reward
def play(latent_dim, datasource, num_actions, num_rewards, encoder, decoder,
         reward_predictor, discriminator, rgb_decoder, transition):

    # Initialize environment
    env = datasource.ZoneIntrudersEnvironment()

    # No-op through the first 3 frames for initial state estimation
    state = env.reset()
    no_op = 3
    s_0, srgb_0 = datasource.convert_frame(state)
    state, reward, done, info = env.step(no_op)
    s_1, srgb_1 = datasource.convert_frame(state)
    state, reward, done, info = env.step(no_op)
    s_2, srgb_2 = datasource.convert_frame(state)
    state_list = [s_0, s_1, s_2]
    rgb_states = np.array([srgb_0, srgb_1, srgb_2])

    # Estimate initial state (given t=0,1,2 estimate state at t=2)
    states = torch.Tensor(state_list).unsqueeze(0)
    z = encoder(states)
    z = transition(z, onehot(no_op))

    #from excitation_bptt import visualize_bptt
    #visualize_bptt(z, transition, reward_predictor, decoder, rgb_decoder, num_actions)

    true_reward = 0
    filename = 'SimpleRolloutAgent-{}.mp4'.format(int(time.time()))
    vid = imutil.Video(filename, framerate=12)
    last_bptt = -25
    t = 2
    humans_killed = 0
    aliens_killed = 0
    while not done:
        z = z.detach()
        # In simulation, compute all possible futures to select the best action
        rewards = []
        for a in range(num_actions):
            z_a = transition(z, onehot(a))
            r_a = compute_rollout_reward(z_a, transition, reward_predictor, num_actions, a)
            rewards.append(r_a)
            #print('Expected reward from taking action {} is {:.03f}'.format(a, r_a))
        max_r = max(rewards)
        max_a = int(np.argmax(rewards))
        #print('Optimal action: {} with reward {:.02f}'.format(max_a, max_r))

        '''
        if t == 20:
            generate_planning_visualization(z, transition, decoder, rgb_decoder, reward_predictor,
                                            num_actions, vid=vid, rollout_width=1, rollout_depth=30,
                                            actions_list = [1, 3, 1, 3, 1] + [3, 3, 3, 1, 3]*5,
                                            caption_title="Neural Simulation")
        '''


        # Take the best action, in real life
        new_state, new_reward, done, info = env.step(max_a)
        humans_killed -= info['total_damage_taken']
        aliens_killed += info['enemy_value_killed']
        true_reward += new_reward

        # Re-estimate state
        ftr_state, rgb_state = datasource.convert_frame(new_state)
        print('t={} curr. r={:.02f} future r: {:.02f} {:.02f} {:.02f} {:.02f}'.format(t, true_reward, rewards[0], rewards[1], rewards[2], rewards[3]))
        caption = 'HUMANS DESTROYED: {}    ALIENS DESTROYED: {}'.format(int(humans_killed), int(aliens_killed))
        print(caption)
        vid.write_frame(rgb_state, resize_to=(512,512), caption=caption)
        imutil.show(rgb_state, resize_to=(512,512), caption=caption, save=False)

        # Decision to shoot
        '''
        if max_a == 1 and last_bptt + 50 < t:
            caption = "Recommended action: FIRE, expected reward: {:.02f}".format(float(max_r))
            for _ in range(10):
                vid.write_frame(rgb_state, resize_to=(512,512), caption=caption)
            from excitation_bptt import visualize_bptt
            visualize_bptt(z, transition, reward_predictor, decoder, rgb_decoder, num_actions, vid)
            last_bptt = t
        '''

        state_list = state_list[1:] + [ftr_state]
        z = encoder(torch.Tensor(state_list).unsqueeze(0))
        z = transition(z, onehot(max_a))
        t += 1
        if t > 300:
            print('Ending evaluation due to time limit')
            break
    vid.finish()
    msg = 'Finished at t={} with cumulative reward {}'.format(t, true_reward)
    with open('evaluation_metrics_{}.txt'.format(int(time.time())), 'w') as fp:
        fp.write(msg + '\n')
    print(msg)


def generate_planning_visualization(z, transition, decoder, rgb_decoder, reward_predictor,
                                    num_actions, vid=None, rollout_width=64, rollout_depth=20,
                                    caption_title="Neural Simulation", actions_list=None):
    if actions_list:
        actions = np.array([actions_list] * rollout_width)
    else:
        actions = np.random.randint(num_actions, size=(rollout_width, rollout_depth))
    cumulative_rewards = torch.zeros(rollout_width).cuda()
    frames = []
    z = z.repeat(rollout_width, 1, 1, 1)
    for t in range(rollout_depth):
        z = transition(z, onehot(actions[:, t]))
        features = decoder(z)
        img = rgb_decoder(features)
        rewards = reward_predictor(z)
        cumulative_rewards += rewards[:, 1] - rewards[:, 0]
        mask = cumulative_rewards + 1
        mask = torch.clamp(mask, 0, 1)
        mask = mask.reshape(-1, 1, 1, 1)
        best_score = float(torch.max(cumulative_rewards))
        caption = "{} t+{} R={:.2f}".format(caption_title, t, best_score)
        img = img * mask
        vid.write_frame(img, resize_to=(512,512), caption=caption)
        frames.append(img)
    for img in frames[::-1]:
        vid.write_frame(img, resize_to=(512,512), caption=caption_title)

    r_max, r_argmax = cumulative_rewards.max(), cumulative_rewards.argmax()
    print('Simulation {} reward: {:.2f}'.format(r_argmax, r_max))


def onehot(a_idx, num_actions=4):
    if type(a_idx) is int:
        # Usage: onehot(2)
        return torch.eye(num_actions)[a_idx].unsqueeze(0).cuda()
    # Usage: onehot([1,2,3])
    return torch.eye(num_actions)[a_idx].cuda()


def compute_rollout_reward(z, transition, reward_predictor, num_actions,
                           selected_action, rollout_width=64, rollout_depth=16,
                           negative_positive_tradeoff=10.0):
    # Initialize a beam
    z = z.repeat(rollout_width, 1, 1, 1)

    # Use 3-step lookahead followed by a no-op rollout policy
    noop_idx = 3
    actions = []
    for i in range(num_actions):
        for j in range(num_actions):
            for k in range(num_actions):
                # Test the 3-action sequence [i,j,k] and then roll out
                actions.append([i, j, k] + [noop_idx] * (rollout_depth - 3))
    actions = torch.LongTensor(np.array(actions)).cuda()
    assert len(actions) == rollout_width

    # Initialize a cumulative reward vector
    cumulative_reward = reward_predictor(z)
    # Starting from z, move forward in time and count the rewards
    for t in range(rollout_depth):
        z = z.detach()
        z = transition(z, onehot(actions[:, t]))
        cumulative_reward += reward_predictor(z)

    # Heuristic, select level of "caution" about negative reward
    cumulative_reward[:, 0] *= negative_positive_tradeoff

    # Average among the rollouts
    max_r, max_idx = cumulative_reward.sum(dim=1).max(dim=0)
    #print('Best possible reward {:.2f} from rollout: {}'.format(max_r, actions[max_idx, :3]))
    return max_r


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


def format_reward_vector(reward):
    return ' '.join(['{:.2f}'.format(r) for r in reward])


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


def visualize_reconstruction(datasource, encoder, decoder, rgb_decoder, transition, reward_predictor, train_iter=0):
    # Image of reconstruction
    filename = 'vis_iter_{:06d}.png'.format(train_iter)
    num_actions = datasource.NUM_ACTIONS
    num_rewards = datasource.NUM_REWARDS
    timesteps = 60
    batch_size = 1
    states, rgb_states, rewards, dones, actions = datasource.get_trajectories(batch_size, timesteps, random_start=False)
    states = torch.Tensor(states).cuda()
    rgb_states = torch.Tensor(rgb_states.transpose(0, 1, 4, 2, 3)).cuda()
    rewards = torch.Tensor(rewards).cuda()
    actions = torch.LongTensor(actions).cuda()
    offsets = [1, 3, 5]
    print('Generating videos for offsets {}'.format(offsets))
    for offset in offsets:
        vid_rgb = imutil.Video('prediction_{:02}_iter_{:06d}.mp4'.format(offset, train_iter), framerate=3)
        vid_reward = imutil.Video('reward_prediction_{:02}_iter_{:06d}.mp4'.format(offset, train_iter), framerate=3)
        vid_aleatoric = imutil.Video('anomaly_detection_{:02}_iter_{:06d}.mp4'.format(offset, train_iter), framerate=3)
        for t in tqdm(range(3, timesteps - offset)):
            # Encode frames t-2, t-1, t to produce state at t-1
            # Then step forward once to produce state at t
            z = encoder(states[:, t-2:t+1])
            z = transition(z, torch.eye(num_actions)[actions[:, t - 1]].cuda())

            # Now step forward *offset* times to produce state at t+offset
            for t_i in range(t, t + offset):
                onehot_a = torch.eye(num_actions)[actions[:, t_i]].cuda()
                z = transition(z, onehot_a)

            # Our prediction of the world from 'offset' steps back
            predicted_features = decoder(z)
            predicted_rgb = rgb_decoder(predicted_features)
            predicted_reward, reward_map = reward_predictor(z, visualize=True)

            # The ground truth
            actual_features = states[:, t + offset]
            actual_rgb = rgb_states[:, t + offset]

            # Difference between actual and predicted outcomes is "surprise"
            surprise_map = torch.clamp((actual_features - predicted_features) ** 2, 0, 1)

            caption = "t={} surprise (aleatoric): {:.03f}".format(t, surprise_map.sum())
            pixels = composite_aleatoric_surprise_image(actual_rgb, surprise_map, z)
            vid_aleatoric.write_frame(pixels, normalize=False, img_padding=8, caption=caption)

            caption = "Left: True t={} Right: Predicted t+{}, Pred. R: {}".format(t, offset, format_reward_vector(predicted_reward[0]))
            pixels = composite_feature_rgb_image(actual_features, actual_rgb, predicted_features, predicted_rgb)
            vid_rgb.write_frame(pixels, normalize=False, img_padding=8, caption=caption)

            caption = "t={} fwd={}, Pred. R: {}".format(t, offset, format_reward_vector(predicted_reward[0]))
            reward_pixels = composite_rgb_reward_factor_image(predicted_rgb, reward_map, z, num_rewards=num_rewards)
            vid_reward.write_frame(reward_pixels, normalize=False, caption=caption)

        vid_rgb.finish()
        vid_reward.finish()
        vid_aleatoric.finish()
    print('Finished generating forward-prediction videos')


def composite_feature_rgb_image(actual_features, actual_rgb, predicted_features, predicted_rgb):
    lbot = imutil.get_pixels(actual_features[0], 384, 512, img_padding=4, normalize=False)
    rbot = imutil.get_pixels(predicted_features[0], 384, 512, img_padding=4, normalize=False)

    height, width, channels = lbot.shape

    ltop = imutil.get_pixels(actual_rgb[0], width, width, normalize=False)
    rtop = imutil.get_pixels(predicted_rgb[0], width, width, normalize=False)

    left = np.concatenate([ltop, lbot], axis=0)
    right = np.concatenate([rtop, rbot], axis=0)

    pixels = np.concatenate([left, right], axis=1)
    pixels = np.clip(pixels, 0, 1)
    return pixels * 255


def composite_rgb_reward_factor_image(x_t_pixels, reward_map, z, num_rewards=4):

    simulated_rgb = imutil.get_pixels(x_t_pixels * 255, 512, 512, normalize=False)

    reward_positive = reward_map[0] * (reward_map[0] > 0).type(torch.cuda.FloatTensor)
    reward_negative = -reward_map[0] * (reward_map[0] < 0).type(torch.cuda.FloatTensor)
    red_map = imutil.get_pixels(reward_negative.sum(dim=0) * 255, 512, 512, normalize=False)
    red_map[:, :, 1:] = 0
    blue_map = imutil.get_pixels(reward_positive.sum(dim=0) * 255, 512, 512, normalize=False)
    blue_map[:, :, :2] = 0

    reward_overlay_simulation = np.clip(simulated_rgb + red_map + blue_map, 0, 255)

    feature_maps = imutil.get_pixels(z[0], 512, 512, img_padding=4) * 255
    composite_visual = np.concatenate([reward_overlay_simulation, feature_maps], axis=1)
    return composite_visual


def composite_aleatoric_surprise_image(x_t_pixels, surprise_map, z, num_factors = 16):
    simulated_rgb = imutil.get_pixels(x_t_pixels * 255, 512, 512, normalize=False)
    # Green for aleatoric surprise
    surprise = surprise_map[0].sum(0) / num_factors
    green_map = imutil.get_pixels(surprise * 255, 512, 512, normalize=False)
    green_map[:, :, 0] = 0
    green_map[:, :, 2] = 0
    #blue_map = imutil.get_pixels(reward_positive.sum(dim=0) * 255, 512, 512, normalize=False)
    #blue_map[:, :, :2] = 0

    reward_overlay_simulation = np.clip(simulated_rgb + green_map, 0, 255)

    feature_maps = imutil.get_pixels(z[0], 512, 512, img_padding=4) * 255
    composite_visual = np.concatenate([reward_overlay_simulation, feature_maps], axis=1)
    return composite_visual


def visualize_forward_simulation(datasource, encoder, decoder, rgb_decoder, transition, reward_pred, train_iter=0, timesteps=60, num_factors=16):
    start_time = time.time()
    print('Starting trajectory simulation for {} frames'.format(timesteps))
    states, rgb_states, rewards, dones, actions = datasource.get_trajectories(batch_size=1, timesteps=timesteps, random_start=False)
    states = torch.Tensor(states).cuda()
    num_actions = datasource.NUM_ACTIONS
    num_rewards = datasource.NUM_REWARDS
    rgb_states = torch.Tensor(rgb_states.transpose(0, 1, 4, 2, 3)).cuda()
    # We begin *at* state t=2, then we simulate from t=2 until t=timesteps
    # Encoder input is t=0, t=1, t=2 to produce t=1
    z = encoder(states[:, :3])
    z = transition(z, torch.eye(num_actions)[actions[:, 1]].cuda())
    z.detach()

    rgb_vid = imutil.Video('simulation_rgb_iter_{:06d}.mp4'.format(train_iter), framerate=3)
    ftr_vid = imutil.Video('simulation_ftr_iter_{:06d}.mp4'.format(train_iter), framerate=3)
    factor_vids = []
    for i in range(num_factors):
        factor_filename = 'factor_videos/separable_factor_{:03d}_iter_{:06d}.mp4'.format(i, train_iter)
        factor_vids.append(imutil.Video(factor_filename, framerate=3))

    # First: replay in simulation the true trajectory
    caption = 'Real'
    simulate_trajectory_from_actions(z.clone(), decoder, rgb_decoder, reward_pred, transition,
                                    states, rgb_states, rewards, dones, actions, rgb_vid, ftr_vid,
                                    factor_vids, caption_tag=caption, num_rewards=num_rewards)

    for vid in factor_vids:
        vid.finish()
    rgb_vid.finish()
    ftr_vid.finish()
    print('Finished trajectory simulation in {:.02f}s'.format(time.time() - start_time))



def simulate_trajectory_from_actions(z, decoder, rgb_decoder, reward_pred, transition,
                                     states, rgb_states, rewards, dones, actions, rgb_vid, ftr_vid, factor_vids,
                                     timesteps=60, caption_tag='', num_actions=4, num_rewards=4):
    estimated_cumulative_reward = np.zeros(num_rewards)
    true_cumulative_reward = np.zeros(num_rewards)
    estimated_rewards = []
    for t in range(2, timesteps - 1):
        x_t, x_t_separable = decoder(z, visualize=True)
        x_t_pixels = rgb_decoder(x_t)
        estimated_reward, reward_map = reward_pred(z, visualize=True)
        estimated_rewards.append(estimated_reward[0])
        estimated_cumulative_reward += estimated_reward[0].data.cpu().numpy()
        true_cumulative_reward += rewards[0, t]

        # Visualize features and RGB
        caption = '{} t+{} a={} R_est={} R_true = {} '.format(caption_tag, t, actions[:, t],
            format_reward_vector(estimated_reward[0]), format_reward_vector(rewards[0, t]))
        rgb_pixels = composite_feature_rgb_image(states[:, t], rgb_states[:, t], x_t, x_t_pixels)
        rgb_vid.write_frame(rgb_pixels, caption=caption, normalize=False)

        # Visualize factors and reward mask
        ftr_pixels = composite_rgb_reward_factor_image(x_t_pixels, reward_map, z)
        ftr_vid.write_frame(ftr_pixels, caption=caption, normalize=False)

        # Visualize each separate factor
        num_factors, num_features, height, width = x_t_separable.shape
        for z_i in range(num_factors):
            factor_vis = rgb_decoder(x_t_separable[z_i].unsqueeze(0), enable_bg=False)
            factor_vids[z_i].write_frame(factor_vis * 255, normalize=False)

        # Predict the next latent point
        onehot_a = torch.eye(num_actions)[actions[:, t]].cuda()
        z = transition(z, onehot_a).detach()

        if dones[0, t]:
            break
    for _ in range(10):
        caption = 'R_est={} R_true = {} '.format(
            format_reward_vector(estimated_cumulative_reward),
            format_reward_vector(true_cumulative_reward))
        rgb_vid.write_frame(rgb_pixels, caption=caption, normalize=False)
        ftr_vid.write_frame(ftr_pixels, caption=caption, normalize=False)
    print('True cumulative reward: {}'.format(format_reward_vector(true_cumulative_reward)))
    print('Estimated cumulative reward: {}'.format(format_reward_vector(estimated_cumulative_reward)))


def measure_prediction_mse(datasource, encoder, decoder, rgb_decoder, transition, reward_pred, train_iter=0, timesteps=100, num_factors=16):
    batch_size = 32
    num_actions = datasource.NUM_ACTIONS
    num_rewards = datasource.NUM_REWARDS
    states, rgb_states, rewards, dones, actions = datasource.get_trajectories(batch_size=batch_size, timesteps=timesteps, random_start=False)
    states = torch.Tensor(states).cuda()
    rgb_states = torch.Tensor(rgb_states.transpose(0, 1, 4, 2, 3)).cuda()
    rewards = torch.Tensor(rewards).cuda()
    dones = torch.Tensor(dones.astype(int)).cuda()

    # We begin *at* state t=2, then we simulate from t=2 until t=timesteps
    # Encoder input is t=0, t=1, t=2 to produce t=1
    z = encoder(states[:, :3])
    z = transition(z, torch.eye(num_actions)[actions[:, 1]].cuda())
    z.detach()

    # Simulate the future, compare with reality
    mse_losses = []
    active_mask = torch.ones(batch_size).cuda()
    for t in range(2, timesteps):
        active_mask = active_mask * (1 - dones[:, t])
        expected = states[:, t]
        predicted = decoder(z)
        diffs = active_mask * ((expected - predicted)**2).mean(dim=-1).mean(dim=-1).mean(dim=-1)
        rec_loss = torch.mean(diffs) * batch_size / torch.sum(active_mask)
        print('MSE t={} {:.04f}\n'.format(t, rec_loss))
        mse_losses.append(float(rec_loss))

        mae_loss = torch.mean(torch.abs(expected - predicted))
        print('MAE t={} {:.04f}\n'.format(t, mae_loss))
        mae_losses.append(float(mae_loss))
        z = transition(z, torch.eye(num_actions)[actions[:, t]].cuda())

    print('Finished trajectory simulation in {:.02f}s'.format(time.time() - start_time))


if __name__ == '__main__':
    main()
