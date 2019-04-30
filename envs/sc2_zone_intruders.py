import sys
import math
import random
import time
import numpy as np
from threading import Thread

import imutil
import gym

from sc2env.environments.zone_intruders import ZoneIntrudersEnvironment

REPLAY_BUFFER_LEN = 50
MIN_REPLAY_BUFFER_LEN = 4
MAX_TRAJECTORY_LEN = 150
MAX_EPISODES_PER_ENVIRONMENT = 500
NUM_ACTIONS = 4
NUM_REWARDS = 2
NO_OP_ACTION = 0
RGB_SIZE = 256

replay_buffer = []
initialized = False
simulation_iters = 0
env = None
policy = None
sim_thread = None


def init():
    global env
    global initialized
    global sim_thread
    env = ZoneIntrudersEnvironment()
    sim_thread = Thread(target=play_game_thread)
    sim_thread.daemon = True  # hack to kill on ctrl+C
    sim_thread.start()
    initialized = True


def play_game_thread():
    global env
    while True:
        simulate_to_replay_buffer(1)
        if simulation_iters % 100 == 1:
            print('\nSimulator thread has simulated {} trajectories. Replay buffer size is {}'.format(
                simulation_iters, len(replay_buffer)))
        if simulation_iters > 0 and simulation_iters % MAX_EPISODES_PER_ENVIRONMENT == 0:
            del env
            init()


def default_policy(*args, **kwargs):
    return env.action_space.sample()


# Simulate batch_size episodes and add them to the replay buffer
def simulate_to_replay_buffer(batch_size):
    global simulation_iters
    global env
    global policy
    if policy is None:
        policy = default_policy
    for _ in range(batch_size):
        play_episode(env, policy)
        simulation_iters += 1


def play_episode(env, policy):
    states, rgb_states, rewards, actions = [], [], [], []
    state = env.reset()
    reward = np.zeros(NUM_REWARDS)
    done = False
    while True:
        action = policy(state)
        state, rgb_state = convert_frame(state)
        states.append(state)
        rgb_states.append(rgb_state)
        rewards.append(reward)
        actions.append(action)
        if len(states) >= MAX_TRAJECTORY_LEN:
            done = True
        if done:
            break
        state, reward_sum, _, info = env.step(action)
        reward = np.array(list(info.values()))
    trajectory = (np.array(states), np.array(rgb_states), np.array(rewards), np.array(actions))
    add_to_replay_buffer(trajectory)


def add_to_replay_buffer(episode):
    if len(replay_buffer) < REPLAY_BUFFER_LEN:
        replay_buffer.append(episode)
    else:
        idx = np.random.randint(0, REPLAY_BUFFER_LEN)
        replay_buffer[idx] = episode


def get_trajectories(batch_size=8, timesteps=10, random_start=True):
    if not initialized:
        init()

    if not sim_thread.is_alive():
        print('Error: Simulator thread has died!')
        raise Exception('Simulator thread crashed')

    # Run the game and add new episodes into the replay buffer
    while len(replay_buffer) < MIN_REPLAY_BUFFER_LEN:
        print('Waiting for replay buffer to fill, buffer size {}/{}...'.format(
            len(replay_buffer), MIN_REPLAY_BUFFER_LEN))
        time.sleep(1)

    # Sample episodes from the replay buffer
    states_batch, rgb_states_batch, rewards_batch, dones_batch, actions_batch = [], [], [], [], []
    for batch_idx in range(batch_size):
        # Accumulate trajectory clips
        states, rgb_states, rewards, actions = [], [], [], []
        timesteps_remaining = timesteps
        dones = []
        while timesteps_remaining > 0:
            selected_states, selected_rgb_states, selected_rewards, selected_actions = random.choice(replay_buffer)
            if random_start:
                start_idx = np.random.randint(0, len(selected_states) - 3)
            else:
                start_idx = 0
            end_idx = min(start_idx + timesteps_remaining, len(selected_states) - 1)
            duration = end_idx - start_idx
            states.extend(selected_states[start_idx:end_idx])
            rgb_states.extend(selected_rgb_states[start_idx:end_idx])
            rewards.extend(selected_rewards[start_idx:end_idx])
            actions.extend(selected_actions[start_idx:end_idx])
            dones.extend([False for _ in range(duration - 1)] + [True])
            timesteps_remaining -= duration

        # feature_map, rgb_map, reward_vec, done_vec, actions_vec
        states_batch.append(np.array(states))  # BHWC
        rgb_states_batch.append(np.array(rgb_states))
        rewards_batch.append(np.array(rewards))
        dones_batch.append(np.array(dones))
        actions_batch.append(np.array(actions))
    return np.array(states_batch), np.array(rewards_batch), np.array(dones_batch), np.array(actions_batch)


def convert_frame(state, width=64, height=64):
    feature_map, feature_screen, rgb_map, rgb_screen = state
    rgb_screen = rgb_screen.transpose(2, 0, 1)
    return feature_screen, rgb_screen / 255.


if __name__ == '__main__':
    env = ZoneIntrudersEnvironment()
    while True:
        simulate_to_replay_buffer(1)
        import pdb; pdb.set_trace()
