import time
import numpy as np
import random

from tqdm import tqdm

from sc2env.environments.simple_towers import SimpleTowersEnvironment

GAME_SIZE = 64
dataset = None

def init():
    global dataset
    dataset = build_dataset()


def build_dataset(num_actions=4, size=50000):
    dataset = []
    env = SimpleTowersEnvironment()
    for i in tqdm(range(size)):
        state_before = env.reset()
        action_idx = np.random.randint(num_actions)
        action = np.zeros(shape=(num_actions,))
        action[action_idx] = 1.
        state_after, reward, done, info = env.step(action_idx)
        dataset.append((state_before, action, state_after))
    return dataset  # list of tuples


def get_batch(size=32):
    idx = np.random.randint(len(dataset) - size)
    inputs, actions, targets = zip(*dataset[idx:idx + size])
    return inputs, actions, targets


def simulator(factor_batch):
    pass
