# A variational autoencoder with some structural causal model stuff
# Uses a fake environment that looks sort of like Pong
import time
import os
import json
import numpy as np
import random

from tqdm import tqdm
import imutil
from logutil import TimeSeries

GAME_SIZE = 40

class MinipongEnv():
    def __init__(self):
        self.left_y = np.random.randint(10, 30)
        self.right_y = np.random.randint(10, 30)
        self.ball_x = np.random.randint(2, 38)
        self.ball_y = np.random.randint(2, 38)
        self.state = build_state(self.left_y, self.right_y, self.ball_x, self.ball_y)

    # The agent can press one of four buttons
    def step(self, a):
        # Some dimensions change in reaction to the agent's actions
        if a[0]:
            self.right_y -= 3
        elif a[1]:
            self.right_y += 3

        if a[2]:
            self.left_y -= 3
        elif a[3]:
            self.left_y += 3
        # Other dimensions change, but not based on agent actions
        self.ball_x += 1
        #self.ball_y += 1
        self.state = build_state(self.left_y, self.right_y, self.ball_x, self.ball_y)


def build_state(left_y, right_y, ball_x, ball_y):
    state = np.zeros((GAME_SIZE, GAME_SIZE))
    paddle_width = 1
    paddle_height = 4
    ball_size = 1
    left_x = 4
    right_x = GAME_SIZE - 4
    state[left_y - paddle_height:left_y + paddle_height,
               left_x - paddle_width: left_x + paddle_width] = 1.0
    state[right_y - paddle_height:right_y + paddle_height,
               right_x - paddle_width: right_x + paddle_width] = 1.0
    state[ball_y-ball_size:ball_y+ball_size,
               ball_x-ball_size:ball_x+ball_size] = 1.0
    return state


def build_dataset(num_actions, size=50000):
    dataset = []
    for i in tqdm(range(size)):
        env = MinipongEnv()
        before = np.array(env.state)
        action = np.zeros(shape=(num_actions,))
        action[np.random.randint(num_actions)] = 1.
        env.step(action)
        after = np.array(env.state)
        dataset.append((before, action, after))
    return dataset  # list of tuples


def get_batch(dataset, size=32):
    idx = np.random.randint(len(dataset) - size)
    inputs, actions, targets = zip(*dataset[idx:idx + size])
    input_tensor = torch.Tensor(inputs).cuda()
    action_tensor = torch.Tensor(actions).cuda()
    target_tensor = torch.Tensor(targets).cuda()
    return input_tensor, action_tensor, target_tensor


# Continuous inputs are in the range [0, 1] for each dimension
def generate_image_continuous(factors):
    left_y = int(factors[0] * 20) + 10
    right_y = int(factors[1] * 20) + 10
    ball_x = int(factors[2] * 38) + 2
    ball_y = int(factors[3] * 38) + 2
    return build_state(left_y, right_y, ball_x, ball_y)


def simulator(factor_batch):
    batch_size = len(factor_batch)
    images = []
    for i in range(batch_size):
        images.append(generate_image_continuous(factor_batch[i]))
    return np.array(images)
