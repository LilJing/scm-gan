import time
import numpy as np
import random

from tqdm import tqdm
from skimage.draw import polygon

GAME_SIZE = 64
dataset = None

def init():
    global dataset
    dataset = build_dataset()


class BoxesEnv():
    def __init__(self):
        self.width = np.random.uniform(5, 10)
        self.height = self.width
        self.x = np.random.randint(20, 46)
        self.y = np.random.randint(20, 46)
        self.rotation = np.random.randint(0, 90)
        self.state = build_state(self.width, self.height, self.x, self.y, self.rotation)

    # The agent can press one of four buttons
    def step(self, a):
        # Some dimensions change in reaction to the agent's actions
        if a[0]:
            self.x -= 3
        elif a[1]:
            self.x += 3

        if a[2]:
            self.y -= 3
        elif a[3]:
            self.y += 3
        # Other dimensions change, but not based on agent actions
        self.rotation += 3
        self.state = build_state(self.width, self.height, self.x, self.y, self.rotation)


# Continuous inputs are in the range [0, 1] for each dimension
def generate_image_continuous(factors):
    width = int(factors[0] * 15) + 5
    height = int(factors[1] * 15) + 5
    x = int(factors[2] * 26) + 20
    y = int(factors[3] * 26) + 20
    rotation = int(factors[4] * 90)
    return build_state(width, height, x, y, rotation)


def build_state(width, height, x, y, rotation):
    state = np.zeros((GAME_SIZE, GAME_SIZE))
    def polar_to_euc(r, theta):
        return (y + r * np.sin(theta), x + r * np.cos(theta))
    points = [polar_to_euc(width, rotation + t) for t in [
        np.pi/4, 3*np.pi/4, 5*np.pi/4, 7*np.pi/4]]
    r, c = zip(*points)
    rr, cc = polygon(r, c)
    state[rr, cc] = 1.
    return state


def build_dataset(num_actions=4, size=100000):
    dataset = []
    for i in tqdm(range(size)):
        env = BoxesEnv()
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


def simulator(factor_batch):
    batch_size = len(factor_batch)
    images = []
    for i in range(batch_size):
        images.append(generate_image_continuous(factor_batch[i]))
    return np.array(images)
