
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
        #self.state = np.random.randint(2, size=4)
        self.state = np.random.uniform(-1, 1, size=4)

    # The agent takes a binary action (0 or 1) which changes the state
    def step(self, a):
        assert 0 <= a <= 1

        # Bits 0, 1, 2, 3: W X Y Z
        self.state[0], self.state[1], self.state[2], self.state[3] = \
            a, self.state[0], self.state[1], self.state[2]
        #self.state = np.clip(self.state, -1, 1)


def build_dataset(size=10000):
    dataset = []
    for i in tqdm(range(size)):
        env = FourBitsEnv()
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
        self.fc1 = nn.Linear(4 + 1, 16, bias=False)
        self.fc2 = nn.Linear(16, 4, bias=False)
        # one-layer version
        #self.fc1 = nn.Linear(5, 4)

    def forward(self, x):
        x = self.fc1(x)
        x = F.leaky_relu(x, 0.2)
        x = self.fc2(x)
        x = F.tanh(x)
        return x

# This function identifies the non-pruned connections
def compute_causal_graph(model):
    input_size = 5
    rows = []
    for i in range(5):
        x = torch.zeros(5)
        zero_return = torch.matmul(model.fc2.weight.abs(), torch.matmul(model.fc1.weight.abs(), x))# + model.fc2.bias
        x[i] = 1.
        # Continuous version
        one_return = torch.matmul(model.fc2.weight.abs(), torch.matmul(model.fc1.weight.abs(), x))# + model.fc2.bias
        # One-layer version
        #result = torch.matmul(model.fc1.weight.abs(), x)
        # Thresholded version
        def binarize(W, theta=.01):
            return (W.abs() > theta).type(torch.FloatTensor)
        #result = torch.matmul(binarize(model.fc2.weight), torch.matmul(binarize(model.fc1.weight), x))
        rows.append(np.array((one_return - zero_return).abs().cpu().data))
    scm = np.array(rows)
    #scm -= scm.min()
    eps = .0001
    return scm / (scm.max() + eps)
    #return scm


# Simpler than the causal graph, least squares regression
import pandas as pd
def ordinary_least_squares():
    inputs, targets = get_batch(data, size=5000)
    predictions = model(inputs)
    df = pd.concat([pd.DataFrame(np.array(inputs)), pd.DataFrame(np.array(targets))], axis=1)
    df.columns = ['w', 'x', 'y', 'z', 'a', 'wn', 'xn', 'yn', 'zn']

    from statsmodels.api import OLS
    ols = OLS(df['a'], df[['wn', 'xn', 'yn', 'zn']]).fit()
    print("Which dimensions of the state space correlate with our actions?")
    print(ols.summary())


# Just a helper function to render the graph to an image
def render_causal_graph(scm):
    # Headless matplotlib fix
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    import networkx as nx

    plt.cla()
    #G = nx.generators.directed.random_k_out_graph(10, 3, 0.5)
    # Make the scm into a square adjacency matrix
    adjacency = np.zeros((5,5))
    adjacency[:4,:] = scm.transpose()
    adjacency = adjacency.transpose()
    print(adjacency)
    edge_alphas = adjacency.flatten() **2

    from networkx.classes.multidigraph import DiGraph
    G = DiGraph(np.ones((5,5)))

    pos = nx.layout.circular_layout(G)

    node_sizes = [10 for i in range(len(G))]
    M = G.number_of_edges()
    #edge_colors = range(2, M + 2)
    #edge_alphas = [(5 + i) / (M + 4) for i in range(M)]
    #edge_colors = [2 for i in range(len(G))]

    nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='blue')
    edges = nx.draw_networkx_edges(G, pos, node_size=node_sizes, arrowstyle='->', arrowsize=20, edge_cmap=plt.cm.Blues, width=2)
    labels = ['$z_0$', '$z_1$', '$z_2$', '$z_3$', 'a']
    labels = {i: labels[i] for i in range(len(labels))}
    pos = {k: (v[0], v[1] + .1) for (k,v) in pos.items()}
    nx.draw_networkx_labels(G, pos, labels, font_size=16)
    # set alpha value for each edge
    for i in range(M):
        edges[i].set_alpha(edge_alphas[i])
        ax = plt.gca()
        ax.set_axis_off()
        plt.show()
    return imutil.show(plt, return_pixels=True, display=False, save=False)


# ok now, can the network learn the task?
data = build_dataset()
model = Net()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
iters = 100 * 1000
ts = TimeSeries('Training', iters)

vid = imutil.VideoMaker('causal_model.mp4')
for i in range(iters):
    model.zero_grad()
    input, target = get_batch(data)
    predicted = model(input)
    #pred_loss = F.binary_cross_entropy(predicted, target)
    pred_loss = torch.sum((predicted - target) ** 2)
    ts.collect('Prediction loss', pred_loss)

    l1_loss = 0.
    #l1_loss += 2.0 * F.l1_loss(model.fc1.weight, torch.zeros(model.fc1.weight.shape))
    #l1_loss += 2.0 * F.l1_loss(model.fc2.weight, torch.zeros(model.fc2.weight.shape))
    ts.collect('Sparsity loss', l1_loss)

    loss = pred_loss + l1_loss

    loss.backward()
    if i % 1000 == 0:
        print(compute_causal_graph(model))
        scm = compute_causal_graph(model)
        caption = 'Prediction Loss {:.03f}'.format(pred_loss)
        vid.write_frame(render_causal_graph(scm), caption=caption)
        print(model.fc1.weight)
        print(model.fc2.weight)
        print(model.fc2.bias)
    ts.print_every(1)
    optimizer.step()
vid.finish()

print(ts)

scm = compute_causal_graph(model)
imutil.show(render_causal_graph(scm))
