import time
import numpy as np
import random

from tqdm import tqdm
from multi_env import MultiEnvironment
from gym.spaces.discrete import Discrete

GAME_SIZE = 64
dataset = None

def init():
    global dataset
    dataset = build_dataset()


class BoxesEnv():
    def __init__(self):
        self.reset()
        self.action_space = Discrete(4)

    # The agent can press one of four buttons
    def step(self, a):
        # Some dimensions change in reaction to the agent's actions
        if a == 0:
            self.x -= 3
        elif a == 1:
            self.x += 3

        if a == 2:
            self.y -= 3
        elif a == 3:
            self.y += 3

        self.x %= GAME_SIZE
        self.y %= GAME_SIZE
        self.state = build_state(self.width, self.height, self.x, self.y)
        # Reward is zero and the game never ends
        return self.state, 0, False, {}

    def reset(self):
        self.width = np.random.uniform(5, 10)
        self.height = np.random.randint(5, 10)
        self.x = np.random.randint(0 + 4, GAME_SIZE - 4)
        self.y = np.random.randint(0 + 4, GAME_SIZE - 4)
        self.state = build_state(self.width, self.height, self.x, self.y)


# Continuous inputs are in the range [0, 1] for each dimension
def generate_image_continuous(factors):
    width = int(factors[0] * 5) + 5
    height = int(factors[1] * 5) + 5
    x = int(factors[2] * GAME_SIZE)
    y = int(factors[3] * GAME_SIZE)
    return build_state(width, height, x, y)


def build_state(width, height, x, y):
    state = np.zeros((GAME_SIZE, GAME_SIZE))
    y0 = int(y - height)
    y1 = int(y + height)
    x0 = int(x - width)
    x1 = int(x + width)
    y0 = np.clip(y0, 0, GAME_SIZE - 1)
    y1 = np.clip(y1, 0, GAME_SIZE - 1)
    x0 = np.clip(x0, 0, GAME_SIZE - 1)
    x1 = np.clip(x1, 0, GAME_SIZE - 1)
    state[y0:y1, x0:x1] = 1.0
    return state


def build_dataset(num_actions=4, size=100000):
    dataset = []
    for i in tqdm(range(size)):
        env = BoxesEnv()
        before = np.array(env.state)
        action = np.zeros(shape=(num_actions,))
        action[np.random.randint(num_actions)] = 1.
        action = np.random.randint(num_actions)
        env.step(action)
        after = np.array(env.state)
        dataset.append((before, action, after))
    return dataset  # list of tuples


def get_batch(size=32):
    idx = np.random.randint(len(dataset) - size)
    inputs, actions, targets = zip(*dataset[idx:idx + size])
    return inputs, actions, targets


def simulator(factor_batch):
    batch_size = len(factor_batch)
    images = []
    for i in range(batch_size):
        images.append(generate_image_continuous(factor_batch[i]))
    return np.array(images)


def get_trajectories(batch_size=32, timesteps=10, policy=None):
    envs = MultiEnvironment([BoxesEnv() for _ in range(batch_size)])
    t_states, t_rewards, t_dones, t_actions = [], [], [], []
    for t in range(timesteps):
        actions = np.random.randint(envs.action_space.n, size=(batch_size,))
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
