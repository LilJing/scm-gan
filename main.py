import argparse

import time
import math
import os
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import imutil
from logutil import TimeSeries
import pandas as pd

import models
from datasource import allocate_datasource
from causal_graph import render_causal_graph
from higgins import higgins_metric_conv
from utils import cov


parser = argparse.ArgumentParser(description="Learn to model a sequential environment")
parser.add_argument('--env', required=True, help='One of: boxes, minipong, Pong-v0, etc (see envs/ for list)')
parser.add_argument('--load-from', type=str, help='Directory containing .pth models to load before starting')
parser.add_argument('--evaluate', action='store_true', help='If true, evaluate instead of training')
parser.add_argument('--evaluations', type=int, default=1, help='Integer number of evaluations to run')
parser.add_argument('--title', type=str, help='Name of experiment in output figures')
parser.add_argument('--batch-size', type=int, default=32, help='Training batch size')
parser.add_argument('--train-iters', type=int, default=10000, help='Number of iterations of training')

parser.add_argument('--truncate-bptt', action='store_true', help='Train only with timestep-local information (training only)')
parser.add_argument('--latent-overshooting', action='store_true', help='Train with Latent Overshooting from Hafner et al. (training only)')
parser.add_argument('--latent-td', action='store_true', help='Train with the Temporal Difference objective (training only)')
parser.add_argument('--td-lambda', type=float, default=0.9, help='Scalar lambda hyperparameter for TD or overshooting (training only)')
parser.add_argument('--td-steps', type=int, default=3, help='Number of concurrent TD forward predictions (training only)')
parser.add_argument('--horizon-min', type=int, default=3, help='Min timestep horizon value (training only)')
parser.add_argument('--horizon-max', type=int, default=10, help='Max timestep horizon value (training only)')
parser.add_argument('--learning-rate', type=float, default=.001, help='Adam lr value (training only)')
parser.add_argument('--finetune-reward', action='store_true', help='Train ONLY the reward estimation network (training only)')
parser.add_argument('--reward-coef', type=float, default=.001, help='Reward loss magnitude (training only)')
parser.add_argument('--activation-l1-coef', type=float, default=.01, help='Activation sparsity coefficient (training only)')
parser.add_argument('--transition-l1-coef', type=float, default=.01, help='Transition sparsity coefficient (training only)')
args = parser.parse_args()




def main():
    latent_dim = 16

    datasource = allocate_datasource(args.env)
    num_actions = datasource.binary_input_channels
    num_rewards = datasource.scalar_output_channels
    input_channels = datasource.conv_input_channels
    output_channels = datasource.conv_output_channels

    encoder = (models.Encoder(latent_dim, input_channels))
    decoder = (models.Decoder(latent_dim, output_channels))
    reward_predictor = (models.RewardPredictor(latent_dim, num_rewards))
    discriminator = (models.Discriminator())
    transition = (models.Transition(latent_dim, num_actions))

    if args.load_from is None:
        print('No --load-from directory specified: initializing new networks')
    elif 'model-encoder.pth' not in os.listdir(args.load_from):
        print('Error: Failed to load saved models from directory {}'.format(args.load_from))
        raise ValueError('Failed to load weights from *.pth')
    else:
        print('Loading models from directory {}'.format(args.load_from))
        encoder.load_state_dict(torch.load(os.path.join(args.load_from, 'model-encoder.pth')))
        decoder.load_state_dict(torch.load(os.path.join(args.load_from, 'model-decoder.pth')))
        transition.load_state_dict(torch.load(os.path.join(args.load_from, 'model-transition.pth')))
        discriminator.load_state_dict(torch.load(os.path.join(args.load_from, 'model-discriminator.pth')))
        reward_predictor.load_state_dict(torch.load(os.path.join(args.load_from, 'model-reward_predictor.pth')))

    if args.evaluate:
        evaluate(datasource, encoder, decoder, transition, discriminator, reward_predictor, latent_dim, use_training_set=True)
        for _ in range(args.evaluations):
            play(latent_dim, datasource, num_actions, num_rewards, encoder, decoder,
                 reward_predictor, discriminator, transition)
        print('Finished {} evaluations'.format(args.evaluations))
    else:
        train(latent_dim, datasource, num_actions, num_rewards, encoder, decoder,
              reward_predictor, discriminator, transition)
    print('Finished execution, terminating')


def train(latent_dim, datasource, num_actions, num_rewards,
          encoder, decoder, reward_predictor, discriminator, transition):
    batch_size = args.batch_size
    train_iters = args.train_iters
    td_lambda_coef = args.td_lambda
    td_steps = args.td_steps
    truncate_bptt = args.truncate_bptt
    enable_td = args.latent_td
    enable_latent_overshooting = args.latent_overshooting
    learning_rate = args.learning_rate
    min_prediction_horizon = args.horizon_min
    max_prediction_horizon = args.horizon_max
    finetune_reward = args.finetune_reward
    REWARD_COEF = args.reward_coef
    ACTIVATION_L1_COEF = args.activation_l1_coef
    TRANSITION_L1_COEF = args.transition_l1_coef

    opt_enc = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
    opt_dec = torch.optim.Adam(decoder.parameters(), lr=learning_rate)
    opt_trans = torch.optim.Adam(transition.parameters(), lr=learning_rate)
    opt_disc = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)
    opt_pred = torch.optim.Adam(reward_predictor.parameters(), lr=learning_rate)
    ts = TimeSeries('Training Model', train_iters, tensorboard=True)

    for train_iter in range(0, train_iters + 1):
        if train_iter % 1000 == 0:
            print('Evaluating networks...')
            evaluate(datasource, encoder, decoder, transition, discriminator, reward_predictor, latent_dim, train_iter=train_iter)
            print('Saving networks to filesystem...')
            torch.save(transition.state_dict(), 'model-transition.pth')
            torch.save(encoder.state_dict(), 'model-encoder.pth')
            torch.save(decoder.state_dict(), 'model-decoder.pth')
            torch.save(discriminator.state_dict(), 'model-discriminator.pth')
            torch.save(reward_predictor.state_dict(), 'model-reward_predictor.pth')

        theta = train_iter / train_iters
        pred_delta = max_prediction_horizon - min_prediction_horizon
        prediction_horizon = min_prediction_horizon + int(pred_delta * theta)

        train_mode([encoder, decoder, transition, discriminator, reward_predictor])

        # Train encoder/transition/decoder
        opt_enc.zero_grad()
        opt_dec.zero_grad()
        opt_trans.zero_grad()
        opt_pred.zero_grad()

        states, rewards, dones, actions = datasource.get_trajectories(batch_size, prediction_horizon)
        states = torch.Tensor(states).cuda()
        rewards = torch.Tensor(rewards).cuda()
        dones = torch.Tensor(dones.astype(int)).cuda()

        # Encode the initial state (using the first 3 frames)
        # Given t, t+1, t+2, encoder outputs the state at time t+1
        z = encoder(states[:, 0:3])

        # Keep track of "done" states to stop a training trajectory at the final time step
        active_mask = torch.ones(batch_size).cuda()

        loss = 0
        td_lambda_loss = 0
        lo_loss = 0
        lo_z_set = {}
        td_z_set = {}
        # Given the state encoded at t=2, predict state at t=3, t=4, ...
        for t in range(1, prediction_horizon - 1):
            active_mask = active_mask * (1 - dones[:, t])

            # Predict reward
            expected_reward = reward_predictor(z)
            actual_reward = rewards[:, t]
            reward_difference = torch.mean(torch.mean((expected_reward - actual_reward)**2, dim=1) * active_mask)
            ts.collect('Rd Loss t={}'.format(t), reward_difference)
            loss += theta * REWARD_COEF * reward_difference  # Normalize by height * width

            # Reconstruction loss
            target_pixels = states[:, t]
            predicted = torch.sigmoid(decoder(z))
            rec_loss_batch = decoder_pixel_loss(target_pixels, predicted)

            if truncate_bptt and t > 1:
                z.detach_()

            rec_loss = torch.mean(rec_loss_batch * active_mask)
            ts.collect('Reconstruction t={}'.format(t), rec_loss)
            loss += rec_loss

            # Apply activation L1 loss
            l1_values = z.abs().mean(-1).mean(-1).mean(-1)
            l1_loss = ACTIVATION_L1_COEF * torch.mean(l1_values * active_mask)
            ts.collect('L1 t={}'.format(t), l1_loss)
            loss += theta * l1_loss

            # Predict transition to the next state
            onehot_a = torch.eye(num_actions)[actions[:, t]].cuda()
            new_z = transition(z, onehot_a)
            # Apply transition L1 loss
            t_l1_values = ((new_z - z).abs().mean(-1).mean(-1).mean(-1))
            t_l1_loss = TRANSITION_L1_COEF * torch.mean(t_l1_values * active_mask)
            ts.collect('T-L1 t={}'.format(t), t_l1_loss)
            loss += theta * t_l1_loss
            z = new_z

            if enable_latent_overshooting:
                # Latent Overshooting, Hafner et al.
                lo_z_set[t] = encoder(states[:, t-1:t+2])

                # For each previous t_left, step forward to t
                for t_left in range(1, t):
                    a = torch.eye(num_actions)[actions[:, t - 1]].cuda()
                    lo_z_set[t_left] = transition(lo_z_set[t_left], a)
                for t_a in range(2, t - 1):
                    # It's like TD but only N:1 for all N
                    predicted_activations = lo_z_set[t_a]
                    target_activations = lo_z_set[t].detach()
                    lo_loss_batch = latent_state_loss(target_activations, predicted_activations)
                    lo_loss += td_lambda_coef * torch.mean(lo_loss_batch * active_mask)

            if not enable_td:
                continue

            # TD-Lambda Loss
            # State i|j is the predicted state at time i conditioned on observation j
            # Eg. if we predict 2 steps perfectly, then state 3|1 will be equal to state 3|3
            # Also, state 3|1 will be equal to state 3|2
            td_z_set[t] = encoder(states[:, t-1:t+2])

            # For each previous t_left, step forward to t
            for t_left in range(1, t):
                a = torch.eye(num_actions)[actions[:, t - 1]].cuda()
                td_z_set[t_left] = transition(td_z_set[t_left], a)

            # At time t_r, consider each combination (t_a, t_b) where a < b <= r
            # At t_a, we thought t_r would be s_{r|a}
            # But later at t_b, we updated our belief to s_{r|b}
            # Update s_{r|a} to be closer to s_{r|b}, for every b up to and including s_{r|r}
            for t_a in range(2, t - 1):
                # Single-Step TD: 4:3, 3:2, 2:1
                # Multi-Step TD: 4:3, 4:2, 4:1, 3:2, 3:1...
                for t_b in range(t_a + 1, min(t_a + td_steps, t + 1)):
                    # Learn a guess, from a guess
                    predicted_activations = td_z_set[t_a]
                    target_activations = td_z_set[t_b].detach()
                    td_loss_batch = latent_state_loss(target_activations, predicted_activations)
                    td_loss = torch.mean(td_loss_batch * active_mask)
                    td_coef = td_lambda_coef ** (t_b - t_a - 1) * td_lambda_coef ** (t_a - 1)
                    td_lambda_loss += td_coef * td_loss
                    # TD including reward
                    predicted_r = reward_predictor(predicted_activations)
                    target_r = reward_predictor(target_activations).detach()
                    r_diffs = torch.mean((predicted_r - target_r)**2, dim=1)
                    r_loss = torch.mean(r_diffs * active_mask)
                    td_lambda_loss += td_coef * r_loss
            # end TD time loop
        if enable_latent_overshooting:
            ts.collect('LO total', lo_loss)
            loss += theta * lo_loss
        if enable_td:
            ts.collect('TD total', td_lambda_loss)
            loss += theta * td_lambda_loss
        loss.backward()

        opt_pred.step()
        if not args.finetune_reward:
            opt_enc.step()
            opt_dec.step()
            opt_trans.step()
        ts.print_every(10)
    print(ts)
    print('Finished')


def latent_state_loss(target, predicted):
    # MSE
    return ((target - predicted)**2).mean(-1).mean(-1).mean(-1)
    # BCE
    #eps = .0001
    #target = torch.clamp(target, eps, 1 - eps)
    #rec_loss_batch = F.binary_cross_entropy(predicted, target, reduction='none')
    #return rec_loss_batch.mean(-1).mean(-1).mean(-1)


def decoder_pixel_loss(target, predicted):
    # MSE
    #return ((target - torch.sigmoid(predicted_logits))**2).mean(-1).mean(-1).mean(-1)
    #eps = .0001
    #target = torch.clamp(target, eps, 1 - eps)
    rec_loss_batch = F.binary_cross_entropy(predicted, target, reduction='none')
    #rec_loss_batch = 0.5 * rec_loss_batch + 0.5 * target
    return rec_loss_batch.mean(-1).mean(-1).mean(-1)


def evaluate(datasource, encoder, decoder, transition, discriminator, reward_predictor, latent_dim, train_iter=0, use_training_set=False):
    print('Evaluating networks...')
    test_mode([encoder, decoder, transition, discriminator, reward_predictor])

    timestamp = str(int(time.time()))
    measure_prediction_mse(datasource, encoder, decoder, transition, reward_predictor, train_iter, num_factors=latent_dim, use_training_set=use_training_set)
    visualize_forward_simulation(datasource, encoder, decoder, transition, reward_predictor, train_iter, num_factors=latent_dim)
    visualize_reconstruction(datasource, encoder, decoder, transition, reward_predictor, train_iter=train_iter)


# Apply a simple model-predictive control algorithm using the learned model,
# to take actions that will maximize reward
def play(latent_dim, datasource, num_actions, num_rewards, encoder, decoder,
         reward_predictor, discriminator, transition):

    # Initialize environment
    env = datasource.make_env(screen_size=512)

    # No-op through the first 3 frames for initial state estimation
    state = env.reset()
    no_op = 3
    s_0, _ = datasource.convert_frame(state)
    state, reward, done, info = env.step(no_op)
    s_1, _ = datasource.convert_frame(state)
    state, reward, done, info = env.step(no_op)
    s_2, _ = datasource.convert_frame(state)
    state_list = [s_0, s_1, s_2]

    # Estimate initial state (given t=0,1,2 estimate state at t=2)
    states = torch.Tensor(state_list).cuda().unsqueeze(0)
    z = encoder(states)
    z = transition(z, onehot(no_op))

    cumulative_reward = 0
    filename = 'SimpleRolloutAgent-{}.mp4'.format(int(time.time()))
    vid = imutil.Video(filename, framerate=12)
    t = 2
    cumulative_negative_reward = 0
    cumulative_positive_reward = 0
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
            generate_planning_visualization(z, transition, decoder, reward_predictor,
                                            num_actions, vid=vid, rollout_width=1, rollout_depth=30,
                                            actions_list = [1, 3, 1, 3, 1] + [3, 3, 3, 1, 3]*5,
                                            caption_title="Neural Simulation")
        '''


        # Take the best action, in real life
        new_state, new_reward, done, info = env.step(max_a)

        positive_reward = sum(v for v in info.values() if v > 0)
        negative_reward = sum(v for v in info.values() if v < 0)

        cumulative_positive_reward += positive_reward
        cumulative_negative_reward -= negative_reward
        cumulative_reward += new_reward

        # Re-estimate state
        ftr_state, rgb_state = datasource.convert_frame(new_state)
        print('t={} curr. r={:.02f} future r: {:.02f} {:.02f} {:.02f} {:.02f}'.format(t, cumulative_reward, rewards[0], rewards[1], rewards[2], rewards[3]))
        caption = 'Negative Reward: {}    Positive Reward: {}'.format(int(cumulative_negative_reward), int(cumulative_positive_reward))
        print(caption)
        vid.write_frame(rgb_state, resize_to=(512,512), caption=caption)

        state_list = state_list[1:] + [ftr_state]
        z = encoder(torch.Tensor(state_list).cuda().unsqueeze(0))
        z = transition(z, onehot(max_a))
        t += 1
        if t > 300:
            print('Ending evaluation due to time limit')
            break
    vid.finish()
    msg = 'Finished at t={} with cumulative reward {}'.format(t, cumulative_reward)
    with open('evaluation_metrics_{}.txt'.format(int(time.time())), 'w') as fp:
        fp.write(msg + '\n')
    print(msg)


def generate_trajectory_video(datasource):
    print("Writing example video of datasource {} to file".format(datasource))
    filename = 'example_trajectory.mp4'
    vid = imutil.Video(filename, framerate=10)
    states, rewards, dones, infos = datasource.get_trajectories(batch_size=1)
    for state in states[0]:
        img = state.transpose(1,2,0)
        vid.write_frame(img, resize_to=(256,256))
    vid.finish()


def generate_planning_visualization(z, transition, decoder, reward_predictor,
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
        features = torch.sigmoid(features)
        rewards = reward_predictor(z)
        cumulative_rewards += rewards[:, 1] - rewards[:, 0]
        mask = cumulative_rewards + 1
        mask = torch.clamp(mask, 0, 1)
        mask = mask.reshape(-1, 1, 1, 1)
        best_score = float(torch.max(cumulative_rewards))
        caption = "{} t+{} R={:.2f}".format(caption_title, t, best_score)
        img = features * mask
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
    num_actions = datasource.binary_input_channels
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


def visualize_reconstruction(datasource, encoder, decoder, transition, reward_predictor, train_iter=0):
    num_actions = datasource.binary_input_channels
    num_rewards = datasource.scalar_output_channels
    timesteps = 45
    batch_size = 1
    states, rewards, dones, actions = datasource.get_trajectories(batch_size, timesteps, random_start=False)
    states = torch.Tensor(states).cuda()
    rewards = torch.Tensor(rewards).cuda()
    actions = torch.LongTensor(actions).cuda()
    offsets = [1, 3]
    print('Generating videos for offsets {}'.format(offsets))
    for offset in offsets:
        vid_rgb = imutil.Video('prediction_{:02}_iter_{:06d}.mp4'.format(offset, train_iter), framerate=3)
        vid_aleatoric = imutil.Video('anomaly_detection_{:02}_iter_{:06d}.mp4'.format(offset, train_iter), framerate=3)
        #vid_reward = imutil.Video('reward_prediction_{:02}_iter_{:06d}.mp4'.format(offset, train_iter), framerate=3)
        for t in range(3, timesteps - offset):
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
            predicted_features = torch.sigmoid(predicted_features)
            predicted_rgb = predicted_features
            predicted_reward, reward_map = reward_predictor(z, visualize=True)

            # The ground truth
            actual_features = states[:, t + offset]
            actual_rgb = convert_ndim_image_to_rgb(actual_features)

            # Difference between actual and predicted outcomes is "surprise"
            surprise_map = torch.clamp((actual_features - predicted_features) ** 2, 0, 1)

            caption = "t={} surprise (aleatoric): {:.03f}".format(t, surprise_map.sum())
            pixels = composite_aleatoric_surprise_image(actual_rgb, surprise_map, z)
            vid_aleatoric.write_frame(pixels, normalize=False, img_padding=8, caption=caption)

            caption = "Left: True t={} Right: Predicted t+{}, Pred. R: {}".format(t, offset, format_reward_vector(predicted_reward[0]))
            pixels = composite_feature_rgb_image(actual_features, actual_rgb, predicted_features, predicted_rgb)
            vid_rgb.write_frame(pixels, normalize=False, img_padding=8, caption=caption)

            #caption = "t={} fwd={}, Pred. R: {}".format(t, offset, format_reward_vector(predicted_reward[0]))
            #reward_pixels = composite_rgb_reward_factor_image(predicted_rgb, reward_map, z, num_rewards=num_rewards)
            #vid_reward.write_frame(reward_pixels, normalize=False, caption=caption)

        vid_rgb.finish()
        vid_aleatoric.finish()
        #vid_reward.finish()
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


def visualize_forward_simulation(datasource, encoder, decoder, transition, reward_pred, train_iter=0, timesteps=60, num_factors=16):
    start_time = time.time()
    print('Starting trajectory simulation for {} frames'.format(timesteps))
    states, rewards, dones, actions = datasource.get_trajectories(batch_size=1, timesteps=timesteps, random_start=False)
    states = torch.Tensor(states).cuda()
    num_actions = datasource.binary_input_channels
    num_rewards = datasource.scalar_output_channels
    # rgb_states = torch.Tensor(rgb_states.transpose(0, 1, 4, 2, 3)).cuda()
    # We begin *at* state t=2, then we simulate from t=2 until t=timesteps
    # Encoder input is t=0, t=1, t=2 to produce t=1
    z = encoder(states[:, :3])
    z = transition(z, torch.eye(num_actions)[actions[:, 1]].cuda())
    z.detach()

    ftr_vid = imutil.Video('simulation_ftr_iter_{:06d}.mp4'.format(train_iter), framerate=3)

    # First: replay in simulation the true trajectory
    caption = 'Real'
    simulate_trajectory_from_actions(z.clone(), decoder, reward_pred, transition,
                                    states, rewards, dones, actions, ftr_vid,
                                    caption_tag=caption, num_rewards=num_rewards,
                                    num_actions=num_actions)

    ftr_vid.finish()
    print('Finished trajectory simulation in {:.02f}s'.format(time.time() - start_time))



def simulate_trajectory_from_actions(z, decoder, reward_pred, transition,
                                     states, rewards, dones, actions, ftr_vid,
                                     timesteps=60, caption_tag='', num_actions=4, num_rewards=4):
    estimated_cumulative_reward = np.zeros(num_rewards)
    true_cumulative_reward = np.zeros(num_rewards)
    estimated_rewards = []
    for t in range(2, timesteps - 1):
        x_t, x_t_separable = decoder(z, visualize=True)
        x_t = torch.sigmoid(x_t)
        x_t_pixels = convert_ndim_image_to_rgb(x_t)
        estimated_reward, reward_map = reward_pred(z, visualize=True)
        estimated_rewards.append(estimated_reward[0])
        estimated_cumulative_reward += estimated_reward[0].data.cpu().numpy()
        true_cumulative_reward += rewards[0, t]

        # Visualize features and RGB
        caption = '{} t+{} a={} R_est={} R_true = {} '.format(caption_tag, t, actions[:, t],
            format_reward_vector(estimated_reward[0]), format_reward_vector(rewards[0, t]))
        #rgb_pixels = composite_feature_rgb_image(states[:, t], rgb_states[:, t], x_t, x_t_pixels)
        #rgb_vid.write_frame(rgb_pixels, caption=caption, normalize=False)

        # Visualize factors and reward mask
        ftr_pixels = composite_rgb_reward_factor_image(x_t_pixels, reward_map, z)

        gt_state = states[0, t].mean(0) * 255
        true_pixels = imutil.get_pixels(gt_state, 512, 512, img_padding=8, normalize=False)

        ftr_pixels = np.concatenate([true_pixels, ftr_pixels], axis=1)
        ftr_vid.write_frame(ftr_pixels, caption=caption, normalize=False)

        # Visualize each separate factor
        num_factors, num_features, height, width = x_t_separable.shape
        #for z_i in range(num_factors):
        #    factor_vis = rgb_decoder(x_t_separable[z_i].unsqueeze(0), enable_bg=False)
        #    factor_vids[z_i].write_frame(factor_vis * 255, normalize=False)

        # Predict the next latent point
        onehot_a = torch.eye(num_actions)[actions[:, t]].cuda()
        z = transition(z, onehot_a).detach()

        if dones[0, t]:
            break
    for _ in range(10):
        caption = 'R_est={} R_true = {} '.format(
            format_reward_vector(estimated_cumulative_reward),
            format_reward_vector(true_cumulative_reward))
        #rgb_vid.write_frame(rgb_pixels, caption=caption, normalize=False)
        ftr_vid.write_frame(ftr_pixels, caption=caption, normalize=False)
    print('True cumulative reward: {}'.format(format_reward_vector(true_cumulative_reward)))
    print('Estimated cumulative reward: {}'.format(format_reward_vector(estimated_cumulative_reward)))


def convert_ndim_image_to_rgb(x):
    if x.shape[1] == 3:
        return x
    return x.sum(dim=1).unsqueeze(1).repeat(1,3,1,1)


def measure_prediction_mse(datasource, encoder, decoder, transition, reward_pred,
                           train_iter=0, timesteps=100, num_factors=16, experiment_name='default',
                           use_training_set=False):
    batch_size = 100
    start_time = time.time()
    num_actions = datasource.binary_input_channels
    num_rewards = datasource.scalar_output_channels
    states, rewards, dones, actions = datasource.get_trajectories(batch_size=batch_size, timesteps=timesteps, training=use_training_set)
    states = torch.Tensor(states).cuda()
    rewards = torch.Tensor(rewards).cuda()
    dones = torch.Tensor(dones.astype(int)).cuda()

    # We begin *at* state t=2, then we simulate from t=2 until t=timesteps
    # Encoder input is t=0, t=1, t=2 to produce t=1
    z = encoder(states[:, :3])
    z = transition(z, torch.eye(num_actions)[actions[:, 1]].cuda())
    z.detach()

    # Simulate the future, compare with reality
    mse_losses = []
    mse_stddevs = []
    active_mask = torch.ones(batch_size).cuda()
    for t in range(2, timesteps):
        active_mask = active_mask * (1 - dones[:, t])
        if sum(active_mask) == 0:
            print('Ending simulation at max trajectory length {}'.format(t))
            break
        expected = states[:, t]
        predicted = decoder(z)
        predicted = torch.sigmoid(predicted)
        diffs = active_mask * ((expected - predicted)**2).mean(dim=-1).mean(dim=-1).mean(dim=-1)
        rec_loss = torch.mean(diffs) * batch_size / torch.sum(active_mask)
        rec_std = torch.std(diffs) * batch_size / torch.sum(active_mask)
        print('MSE t={} {:.04f}\n'.format(t, rec_loss))
        mse_losses.append(float(rec_loss))
        mse_stddevs.append(float(rec_std))

        #mae_loss = torch.mean(torch.abs(expected - predicted))
        #print('MAE t={} {:.04f}\n'.format(t, mae_loss))
        #mae_losses.append(float(mae_loss))
        z = transition(z, torch.eye(num_actions)[actions[:, t]].cuda())
        z.detach_()
    if len(mse_losses) == 0:
        print('Degenerate trajectory, skipping MSE calculation')
        return
    print('Avg. MSE loss: {}'.format(np.mean(mse_losses)))
    print('Finished trajectory simulation in {:.02f}s'.format(time.time() - start_time))

    mse_filename = 'mse_{}_iter_{:06d}.json'.format(experiment_name, train_iter)
    with open(mse_filename, 'w') as fp:
        fp.write(json.dumps(mse_losses, indent=2))
    stddev_filename = 'mse_stddev_{}_iter_{:06d}.json'.format(experiment_name, train_iter)
    with open(stddev_filename, 'w') as fp:
        fp.write(json.dumps(mse_stddevs, indent=2))

    plot_params = {
        'title': 'Mean Squared Error Pixel Loss: {}'.format(args.title),
        'grid': True,
    }
    plt = pd.Series(mse_losses).plot(**plot_params)
    plt.set_ylim(ymin=0)
    plt.set_ylabel('Pixel MSE')
    plt.set_xlabel('Prediction horizon (timesteps)')

    plot_mse(plt, mse_filename, stddev_filename)
    plot_mse(plt, mse_filename, stddev_filename, facecolor='#00FF00', edgecolor='#00FF00')

    filename = 'mse_graph_iter_{:06d}.png'.format(train_iter)
    imutil.show(plt, filename=filename)
    from matplotlib import pyplot
    pyplot.close()


def plot_mse(plt, mean_filename, err_filename, facecolor='#BBBBFF', edgecolor='#0000FF'):
    meanvals = np.array(json.load(open(mean_filename)))
    errvals = np.array(json.load(open(err_filename)))

    # Add shaded region to indicate stddev
    x = np.array(range(len(meanvals)))
    plt.plot(x, meanvals, color=edgecolor)
    plt.fill_between(x, meanvals - errvals, meanvals + errvals,
                     alpha=0.2, facecolor=facecolor, edgecolor=edgecolor)


if __name__ == '__main__':
    main()
