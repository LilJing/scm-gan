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
        state0 = self.ale.getScreenRGB2()
        self.ale.act(0)  # No-op
        state1 = self.ale.getScreenRGB2()
        return crop(state0, state1)

    def step(self, action):
        # Hard-code two steps per action, pixel-wise max over two frames
        # Removes flickering
        reward = 0
        states = []
        for _ in range(2):
            if not self.ale.game_over():
                reward += self.ale.act(action)
            states.append(self.ale.getScreenRGB2())
            done = self.ale.game_over()
        state = crop(*states)
        info = {'ale.lives': self.ale.lives()}
        return state, reward, done, info


def crop(state1, state2):
    output = np.zeros((3,96,64))
    for c in range(3):
        pixels = np.maximum(state1, state2)
        output[c] = block_reduce(pixels[24:-34, 16:-16, c], (2, 2), np.max) / 255.
    # Output format is 96x64 with no flickering
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


MAX_BATCH_SIZE = 32
envs = MultiEnvironment([GameEnv() for _ in range(MAX_BATCH_SIZE)])
states = envs.reset()
def get_trajectories(batch_size=32, timesteps=10, policy=None, random_start=False):
    global states
    actions = np.random.randint(envs.action_space.n, size=(batch_size,))

    PolicyFn = policy or HeuristicPolicy
    policies = [PolicyFn() for _ in range(batch_size)]

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
    states = np.swapaxes(t_states, 0, 1)[:batch_size]
    rewards = np.swapaxes(t_rewards, 0, 1)[:batch_size]
    dones = np.swapaxes(t_dones, 0, 1)[:batch_size]
    actions = np.swapaxes(t_actions, 0, 1)[:batch_size]
    return states, rewards, dones, actions



if __name__ == '__main__':
    import imutil
    print('Simulation time benchmark: Centipede')
    print('Simulating {} games for 100 timesteps...'.format(MAX_BATCH_SIZE))
    vid = imutil.Video('centipede.mp4', framerate=5)
    start_time = time.time()
    batches = 10
    timesteps = 10
    for _ in range(batches):
        states, rewards, dones, actions = get_trajectories(timesteps=timesteps)
        for state, action, reward in zip(states[0], actions[0], rewards[0]):
            caption = "Prev. Action {} Prev Reward {}".format(action, reward)
            vid.write_frame(state.transpose(1,2,0), img_padding=8, resize_to=(512,512), caption=caption)
    duration = time.time() - start_time
    print('Finished simulating {} games for {} timesteps in {:.3f} sec'.format(
        MAX_BATCH_SIZE, timesteps*batches, duration))
    vid.finish()
