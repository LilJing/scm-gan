import sys
import math
import random
import time
import numpy as np
from threading import Thread

import imutil
import gym

REPLAY_BUFFER_LEN = 100
MIN_REPLAY_BUFFER_LEN = 4
MAX_TRAJECTORY_LEN = 200
MAX_EPISODES_PER_ENVIRONMENT = 500
BURN_STATES_BEFORE_START = 50
ENV_NAME = 'SpaceInvadersDeterministic-v4'
NUM_ACTIONS = 6
NUM_REWARDS = 1
RGB_SIZE = 64

replay_buffer = []
initialized = False
simulation_iters = 0
env = None
policy = None


def init():
    global env
    global initialized
    env = gym.make(ENV_NAME)
    thread = Thread(target=play_game_thread)
    thread.daemon = True  # hack to kill on ctrl+C
    thread.start()
    initialized = True


def play_game_thread():
    while True:
        simulate_to_replay_buffer(1)


# Simulate batch_size episodes and add them to the replay buffer
def simulate_to_replay_buffer(batch_size):
    global simulation_iters
    global env
    global policy
    def default_policy(*args, **kwargs):
        return env.action_space.sample()
    if policy is None:
        policy = default_policy
    for _ in range(batch_size):
        play_episode(env, policy)
        simulation_iters += 1


def play_episode(env, policy):
    states, rewards, actions = [], [], []
    state = env.reset()
    for _ in range(BURN_STATES_BEFORE_START):
        state, _, _, _ = env.step(action=0)
    reward = [0]
    done = False
    while True:
        action = policy(state)
        state = convert_atari_frame(state)
        states.append(state)
        rewards.append(reward)
        actions.append(action)
        if len(states) >= MAX_TRAJECTORY_LEN:
            done = True
        if done:
            break
        state, reward, done, info = env.step(action)
        reward = [reward]
    trajectory = (np.array(states), np.array(rewards), np.array(actions))
    add_to_replay_buffer(trajectory)


def add_to_replay_buffer(episode):
    if len(replay_buffer) < REPLAY_BUFFER_LEN:
        replay_buffer.append(episode)
    else:
        idx = np.random.randint(1, REPLAY_BUFFER_LEN)
        replay_buffer[idx] = episode


def get_trajectories(batch_size=8, timesteps=10, random_start=True):
    if not initialized:
        init()

    # Run the game and add new episodes into the replay buffer
    while len(replay_buffer) < MIN_REPLAY_BUFFER_LEN:
        print('Waiting for replay buffer to fill, buffer size {}/{}...'.format(
            len(replay_buffer), MIN_REPLAY_BUFFER_LEN))
        time.sleep(1)

    # Sample episodes from the replay buffer
    states_batch, rewards_batch, dones_batch, actions_batch = [], [], [], []
    for batch_idx in range(batch_size):
        # Accumulate trajectory clips
        states, rewards, actions = [], [], []
        timesteps_remaining = timesteps
        dones = []
        while timesteps_remaining > 0:
            selected_states, selected_rewards, selected_actions = random.choice(replay_buffer)
            if random_start:
                start_idx = np.random.randint(0, len(selected_states) - 3)
            else:
                start_idx = 0
            end_idx = min(start_idx + timesteps_remaining, len(selected_states) - 1)
            duration = end_idx - start_idx
            states.extend(selected_states[start_idx:end_idx])
            rewards.extend(selected_rewards[start_idx:end_idx])
            actions.extend(selected_actions[start_idx:end_idx])
            dones.extend([False for _ in range(duration - 1)] + [True])
            timesteps_remaining -= duration

        states_batch.append(np.array(states))  # BHWC
        rewards_batch.append(np.array(rewards))
        dones_batch.append(np.array(dones))
        actions_batch.append(np.array(actions))
    states = np.array(states_batch)
    rgb_states = states
    rewards = np.array(rewards_batch)
    dones = np.array(dones_batch)
    actions = np.array(actions_batch)
    return states, rgb_states, rewards, dones, actions


def convert_atari_frame(state, width=64, height=64):
    if ENV_NAME.startswith('SpaceInvaders'):
        # Crop to playable area
        state = state[20:]

    state = imutil.get_pixels(state, width, height)
    state = state.transpose((2,0,1))
    return state


if __name__ == '__main__':
    ENV_NAME = sys.argv[1]
    start_time = time.time()
    batch_size = 8
    MAX_ITERS = 1000
    for i in range(0, MAX_ITERS, batch_size):
        get_trajectories(batch_size)
        duration = time.time() - start_time
        print('Generated {} trajectories in {:.03f} sec ({:.02f} traj/sec) replay buffer len {}'.format(
            i, duration, i / duration, len(replay_buffer)))
