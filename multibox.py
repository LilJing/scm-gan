import time
import numpy as np
import random

from tqdm import tqdm

GAME_SIZE = 64
dataset = None

def init():
    global dataset
    dataset = build_dataset()


class MultiboxEnv():
    def __init__(self):
        self.red_x = np.random.randint(10, 50)
        self.red_y = np.random.randint(10, 50)
        self.red_radius = np.random.randint(5, 8)
        self.blue_x = np.random.randint(10, 50)
        self.blue_y = np.random.randint(10, 50)
        self.blue_radius = np.random.randint(2, 4)
        self.state = build_state(self.red_x, self.red_y, self.red_radius,
                                 self.blue_x, self.blue_y, self.blue_radius)

    # The agent can press one of four buttons
    def step(self, a):
        # Some dimensions change in reaction to the agent's actions
        if a[0]:
            self.red_x -= 3
        elif a[1]:
            self.red_x += 3

        if a[2]:
            self.red_y -= 3
        elif a[3]:
            self.red_y += 3

        self.state = build_state(self.red_x, self.red_y, self.red_radius,
                                 self.blue_x, self.blue_y, self.blue_radius)


def build_state(red_x, red_y, red_r, blue_x, blue_y, blue_r):
    state = np.zeros((GAME_SIZE, GAME_SIZE))
    state[red_y - red_r:red_y + red_r,
          red_x - red_r:red_x + red_r] = 1.0
    state[blue_y - blue_r:blue_y + blue_r,
          blue_x - blue_r:blue_x + blue_r] = 1.0
    return state


def build_dataset(num_actions=4, size=50000):
    dataset = []
    for i in tqdm(range(size)):
        env = MultiboxEnv()
        before = np.array(env.state)
        action = np.zeros(shape=(num_actions,))
        action[np.random.randint(num_actions)] = 1.
        env.step(action)
        after = np.array(env.state)
        dataset.append((before, action, after))
    return dataset  # list of tuples


def get_batch(size=32):
    idx = np.random.randint(len(dataset) - size)
    inputs, actions, targets = zip(*dataset[idx:idx + size])
    return inputs, actions, targets


# Continuous inputs are in the range [0, 1] for each dimension
def generate_image_continuous(factors):
    red_x = int(factors[0] * 40) + 10
    red_y = int(factors[1] * 40) + 10
    red_r = int(factors[2] * 3) + 5

    blue_x = int(factors[3] * 40) + 10
    blue_y = int(factors[4] * 40) + 10
    blue_r = int(factors[5] * 2) + 2
    return build_state(red_x, red_y, red_r, blue_x, blue_y, blue_r)


def simulator(factor_batch):
    batch_size = len(factor_batch)
    images = []
    for i in range(batch_size):
        images.append(generate_image_continuous(factor_batch[i]))
    return np.array(images)
