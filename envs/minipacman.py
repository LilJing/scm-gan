import sys
import math
import random
import time
import numpy as np
from threading import Thread

import imutil
import gym


REPLAY_BUFFER_LEN = 50
MIN_REPLAY_BUFFER_LEN = 4
MAX_TRAJECTORY_LEN = 150
MAX_EPISODES_PER_ENVIRONMENT = 500
NUM_ACTIONS = 5
NUM_REWARDS = 2
NO_OP_ACTION = 0

replay_buffer_training = []
replay_buffer_testing = []
initialized = False
simulation_iters = 0
env = None
policy = None
sim_thread = None


from gym_minipacman.envs.minipacman_env import MiniPacman, ALE
class MiniPacManEnv(MiniPacman):
    def __init__(self):
        # {0:'NOOP',1:'RIGHT',2:'UP',3:'LEFT',4:'DOWN'}
        self.ale = ALE(1)
        self.step_reward = 0.
        self.food_reward = 1.
        self.big_pill_reward = 2.
        self.ghost_hunt_reward = 5.
        self.ghost_death_reward = -1
        self.all_pill_terminate = False
        self.all_ghosts_terminate = False
        self.all_food_terminate = True
        self.timer_terminate = -1
        super().__init__()


def make_env(*args, **kwargs):
    return MiniPacManEnv()


def init():
    global env
    global initialized
    global sim_thread
    env = make_env()
    sim_thread = Thread(target=play_game_thread)
    sim_thread.daemon = True  # hack to kill on ctrl+C
    sim_thread.start()
    initialized = True


def play_game_thread():
    global env
    while True:
        simulate_to_replay_buffer(1)
        if simulation_iters % 100 == 1:
            print('\nSimulator thread has simulated {} trajectories. Replay buffer len training {}, testing {}'.format(
                simulation_iters, len(replay_buffer_training), len(replay_buffer_testing)))
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
    states, rewards, actions = [], [], []
    state = env.reset()
    reward = np.zeros(NUM_REWARDS)
    done = False
    while True:
        action = policy(state)
        state = convert_frame(state)
        states.append(state)
        rewards.append(reward)
        actions.append(action)
        if len(states) >= MAX_TRAJECTORY_LEN:
            done = True
        if done:
            break
        state, reward_sum, _, info = env.step(action)
        reward[0] = max(0, reward_sum)
        reward[1] = min(0, reward_sum)
    trajectory = (np.array(states), np.array(rewards), np.array(actions))
    add_to_replay_buffer(trajectory)
    time.sleep(1)


def add_to_replay_buffer(episode, test_set_holdout=0.20):
    replay_buffer = replay_buffer_training if np.random.random() > test_set_holdout else replay_buffer_testing

    if len(replay_buffer) < REPLAY_BUFFER_LEN:
        replay_buffer.append(episode)
    else:
        idx = np.random.randint(0, REPLAY_BUFFER_LEN)
        replay_buffer[idx] = episode


def get_trajectories(batch_size=8, timesteps=10, random_start=True, training=True):
    if not initialized:
        init()

    if not sim_thread.is_alive():
        print('Error: Simulator thread has died!')
        raise Exception('Simulator thread crashed')

    replay_buffer = replay_buffer_training if training else replay_buffer_testing

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

        # feature_map, reward_vec, done_vec, actions_vec
        states_batch.append(np.array(states))  # BHWC
        rewards_batch.append(np.array(rewards))
        dones_batch.append(np.array(dones))
        actions_batch.append(np.array(actions))
    return np.array(states_batch), np.array(rewards_batch), np.array(dones_batch), np.array(actions_batch)


def convert_frame(state):
    return state.transpose((2, 0, 1)).copy()

if __name__ == '__main__':
    start_time = time.time()

    env = make_env()
    simulate_to_replay_buffer(1)

    env = make_env()
    batch_size = 8
    vid = imutil.Video('minipacman.mp4', framerate=5)
    states, rewards, dones, actions = get_trajectories(batch_size, random_start=False, timesteps=100)
    i = 0
    for state, reward, done, action in zip(states[0], rewards[0], dones[0], actions[0]):
        caption = "t={} Prev. Action {} Prev Reward {} Done {}".format(i, action, reward, done)
        vid.write_frame(state.transpose(1,2,0), img_padding=8, resize_to=(512,512), caption=caption)
        print('state {}, {}'.format(state.mean(), caption))
        i += 1
    duration = time.time() - start_time
    print('Generated {} trajectories in {:.03f} sec ({:.02f} traj/sec) replay buffer len {}'.format(
        i, duration, i / duration, len(replay_buffer_training)))
    vid.finish()
    print('Finished')
