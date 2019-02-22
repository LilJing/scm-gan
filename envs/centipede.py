import time
import numpy as np
import random

from tqdm import tqdm
from multi_env import MultiEnvironment
from gym.spaces.discrete import Discrete
import atari_py
import imutil
from skimage.measure import block_reduce

NUM_ACTIONS = 6


class GameEnv():
    def __init__(self, name='centipede'):
        path = atari_py.get_game_path(name)
        self.ale = atari_py.ALEInterface()
        self.ale.loadROM(path)
        self.action_space = Discrete(6)

    def reset(self):
        self.ale.reset_game()
        return crop(self.ale.getScreenRGB2())

    def step(self, action):
        reward = self.ale.act(action)
        state = crop(self.ale.getScreenRGB2())
        done = self.ale.game_over()
        info = {'ale.lives': self.ale.lives()}
        if done:
            self.reset()
        return state, reward, done, info


def crop(state):
    output = np.zeros((3,64,64))
    for c in range(3):
        output[c] = block_reduce(state[24:-34, 16:-16, c], (3, 2), np.mean) / 255.
    return output


class HeuristicPolicy():
    def __init__(self, num_actions=6):
        self.num_actions = num_actions
        self.prev_action = np.random.randint(self.num_actions)

    def step(self, state):
        flip = np.random.random()
        if flip > 0.90:
            action = 1  # mash the 'shoot' button
        elif flip > 0.25:
            action = self.prev_action
            self.prev_action = action
        else:
            action = np.random.randint(self.num_actions)
        return action


envs = None
def get_trajectories(batch_size=32, timesteps=10, policy=None, random_start=False):
    global envs
    if envs is None:
        envs = MultiEnvironment([GameEnv() for _ in range(batch_size)])
    actions = np.random.randint(envs.action_space.n, size=(batch_size,))

    PolicyFn = policy or HeuristicPolicy
    policies = [PolicyFn() for _ in range(batch_size)]

    states = envs.reset()
    t_states, t_rewards, t_dones, t_actions = [], [], [], []
    for t in range(timesteps):
        actions = [p.step(s) for p, s in zip(policies, states)]
        actions = actions[:batch_size]  # hack for fixed envs w/ varible batch size
        states, rewards, dones, _ = envs.step(actions)
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
    vid = imutil.Video('centipede.mp4', framerate=5)
    for state, action, reward in zip(states[0], actions[0], rewards[0]):
        caption = "Prev. Action {} Prev Reward {}".format(action, reward)
        vid.write_frame(state.transpose(2,0,1), img_padding=8, resize_to=(512,512), caption=caption)
    vid.finish()
