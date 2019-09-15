import time
import numpy as np
import random

from multi_env import MultiEnvironment
from gym.spaces.discrete import Discrete

CHANNELS = 1
GAME_SIZE = 64
NUM_ACTIONS = 1
NUM_REWARDS = 1
PREHEAT_STEPS = 2


# Conway's Game of Life
# A zero-player game with simple rules.
# Given the right initial configuration, GoL can compute any function
class Env():
    def __init__(self):
        self.reset()
        self.action_space = Discrete(1)

    def reset(self, p=0.5):
        # One row and column of circular padding for toroidal topology
        self.state = np.random.random((GAME_SIZE, GAME_SIZE)) > p
        for _ in range(PREHEAT_STEPS):
            self.step(0)

    def step(self, a):
        # Ignore input actions; Life is a zero-player game
        from scipy.signal import convolve2d

        # Step forward in time
        kernel = np.ones((3, 3))
        kernel[1, 1] = 0
        nbrs_count = convolve2d(self.state, kernel, mode='same', boundary='wrap')
        self.state = ((nbrs_count == 3) | (self.state & (nbrs_count == 2)))
        reward = 0
        done = False
        info = {}
        output_state = np.expand_dims(self.state, 0).astype(np.float)
        return output_state, reward, done, info


def get_trajectories(batch_size=32, timesteps=10, policy='random', random_start=False, training=False):
    envs = MultiEnvironment([Env() for _ in range(batch_size)])
    t_states, t_rewards, t_dones, t_actions = [], [], [], []
    # Initial actions/stats
    actions = np.random.randint(envs.action_space.n, size=(batch_size,))
    for t in range(timesteps):
        states, rewards, dones, _ = envs.step(actions)
        rewards = [rewards]
        actions = np.random.randint(envs.action_space.n, size=(batch_size,))
        t_states.append(states)
        t_rewards.append(rewards)
        t_dones.append(dones)
        t_actions.append(actions)
    # Reshape to (batch_size, timesteps, ...)
    states = np.swapaxes(t_states, 0, 1)
    rewards = np.swapaxes(t_rewards, 0, 1)
    dones = np.swapaxes(t_dones, 0, 1)
    actions = np.swapaxes(t_actions, 0, 1)
    return states, rewards, dones, actions


if __name__ == '__main__':
    states, rewards, dones, actions = get_trajectories(batch_size=1, timesteps=100)
    import imutil
    vid = imutil.Video('gameoflife.mp4', framerate=5)
    for state, action, reward in zip(states[0], actions[0], rewards[0]):
        pixels = np.transpose(state, (1, 2, 0))
        caption = "Prev. Action {} Prev Reward {}".format(action, reward)
        vid.write_frame(pixels, img_padding=8, resize_to=(512,512), caption=caption)
    vid.finish()
