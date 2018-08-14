import os
import json
import numpy as np
import random

from tqdm import tqdm
import imutil
from logutil import TimeSeries


class FourBitsEnv():
    # The environment consists of four bits
    def __init__(self):
        self.state = np.random.randint(2, size=4)

    # The agent takes a binary action (0 or 1) which changes the state
    def step(self, a):
        assert 0 <= a <= 1

        # Bit 0 is random and unaffected by anything else
        self.state[0] = np.random.randint(2)

        # Bit 1 follows bit 3, one time step behind
        self.state[1] = self.state[3]

        # Bit 2 is correlated to itself and the action
        self.state[2] = self.state[2] if np.random.randint(2) else a

        # Bit 3 is determined entirely by the agent's action
        self.state[3] = a

        # A clever agent will learn the following structural causal model:
        # 3 causes 1
        # 0 causes 2
        # A causes 2
        # A causes 3


def build_dataset(size=10000):
    dataset = []
    env = FourBitsEnv()
    for i in tqdm(range(size)):
        before = np.array(env.state)
        action = np.random.randint(2)
        env.step(action)
        after = np.array(env.state)
        dataset.append((before, action, after))
    return dataset  # list of tuples


def get_batch(dataset, size=32):
    idx = np.random.randint(len(dataset) - size)
    inputs, actions, targets = zip(*dataset[idx:idx + size])
    state_act = np.concatenate([np.array(inputs), np.array(actions).reshape(-1, 1)], axis=1)
    input_tensor = torch.Tensor(state_act)
    target_tensor = torch.Tensor(targets)
    return input_tensor, target_tensor


# ok now we build a neural network
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: State + Action
        # Output: State
        self.fc1 = nn.Linear(4 + 1, 32)
        self.fc2 = nn.Linear(32, 4)

    def forward(self, x):
        x = self.fc1(x)
        x = F.leaky_relu(x, 0.2)
        x = self.fc2(x)
        x = F.sigmoid(x)
        return x


# ok now, can the network learn the task?
data = build_dataset()
model = Net()
optimizer = optim.Adam(model.parameters(), lr=0.001)
iters = 20 * 1000
ts = TimeSeries('Training', iters)

for i in range(iters):
    model.zero_grad()
    input, target = get_batch(data)
    predicted = model(input)
    bce = F.binary_cross_entropy(predicted, target)
    ts.collect('BCE loss', bce)
    #loss = torch.sum((predicted - target) ** 2)

    l1_loss = 0.
    for param in model.parameters():
        l1_loss += .1 * F.l1_loss(param, torch.zeros(param.shape))
    ts.collect('L1 reg loss', l1_loss)

    loss = bce + l1_loss

    loss.backward()
    if i % 10000 == 0:
        print(input[:4])
        print(target[:4])
        print(predicted[:4])
    ts.print_every(1)
    optimizer.step()

print(ts)

import pandas as pd

inputs, targets = get_batch(data, size=5000)
predictions = model(inputs)
df = pd.concat([pd.DataFrame(np.array(inputs)), pd.DataFrame(np.array(targets))], axis=1)
df.columns = ['in0', 'in1', 'in2', 'in3', 'a', 'out0', 'out1', 'out2', 'out3']

from statsmodels.api import OLS
ols = OLS(df['in0'], df[['out0', 'out1', 'out2', 'out3']]).fit()

import pdb; pdb.set_trace()

