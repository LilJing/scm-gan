import time
import os
import json
import numpy as np
import random

from tqdm import tqdm
import imutil
from logutil import TimeSeries


class FallingBoxEnv():
    def __init__(self):
        # The "true" latent space is two values which vary
        self.x = np.random.randint(8, 24)
        self.y = np.random.randint(8, 24)
        self.rotation = np.random.uniform(0, 2 * np.pi)
        self.color = 1.0 # np.random.uniform(0.25, 1.0)
        self.radius = np.random.randint(4, 8)
        self.build_state()

    # The agent can press four buttons: left, right, rotate-left, rotate-right
    def step(self, a):
        # Some dimensions change in reaction to the agent's actions
        if a[0]:
            self.x -= 3
        elif a[1]:
            self.x += 3

        if a[2]:
            self.radius -= 1
        elif a[3]:
            self.radius += 1

        # Other dimensions change, but not based on agent actions
        self.y += 5

        #self.color += .1
        self.build_state()

    def build_state(self):
        self.state = np.zeros((32,32))
        self.state[self.y-self.radius:self.y + self.radius, self.x-self.radius:self.x+self.radius] = self.color
        self.state = np.clip(self.state, 0, 1)


def build_dataset(num_actions, size=10000):
    dataset = []
    for i in tqdm(range(size)):
        env = FallingBoxEnv()
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


# ok now we build a neural network
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Encoder(nn.Module):
    def __init__(self, latent_size):
        super().__init__()
        self.latent_size = latent_size
        # 1x32x32
        self.conv1 = nn.Conv2d(1, 32, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        # 32x32x32
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        # 64x16x16
        self.conv3 = nn.Conv2d(64, 64, 4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        # 64x8x8
        self.conv4 = nn.Conv2d(64, 64, 4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        # 64x4x4
        self.fc1 = nn.Linear(64*4*4, latent_size)
        self.cuda()

    def forward(self, x):
        # Input: batch x 32 x 32
        x = x.unsqueeze(1)
        # batch x 1 x 32 x 32
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x, 0.2)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x, 0.2)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.leaky_relu(x, 0.2)

        x = self.conv4(x)
        x = self.bn4(x)
        x = F.leaky_relu(x, 0.2)

        x = x.view(-1, 64*4*4)
        x = self.fc1(x)
        return x


class Decoder(nn.Module):
    def __init__(self, latent_size):
        super().__init__()
        self.latent_size = latent_size
        self.fc1 = nn.Linear(latent_size, latent_size*2)
        self.deconv1 = nn.ConvTranspose2d(latent_size*2, 128, 4, stride=1)
        self.bn1 = nn.BatchNorm2d(128)
        # 128 x 4 x 4
        self.deconv2 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        # 64 x 8 x 8
        self.deconv3 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        # 32 x 16 x 16
        self.deconv4 = nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(16)
        # 16 x 32 x 32
        self.deconv5 = nn.ConvTranspose2d(16, 1, 3, stride=1, padding=1)
        self.cuda()

    def forward(self, x):
        x = self.fc1(x)
        x = F.leaky_relu(x, 0.2)

        x = x.unsqueeze(-1).unsqueeze(-1)
        x = self.deconv1(x)
        x = F.leaky_relu(x, 0.2)
        x = self.bn1(x)

        x = self.deconv2(x)
        x = F.leaky_relu(x, 0.2)
        x = self.bn2(x)

        x = self.deconv3(x)
        x = F.leaky_relu(x, 0.2)
        x = self.bn3(x)

        x = self.deconv4(x)
        x = F.leaky_relu(x, 0.2)
        x = self.bn4(x)

        x = self.deconv5(x)
        x = F.sigmoid(x)
        return x



class Transition(nn.Module):
    def __init__(self, latent_size, num_actions):
        super().__init__()
        # Input: State + Action
        # Output: State
        self.fc1 = nn.Linear(latent_size + num_actions, 16, bias=False)
        self.fc2 = nn.Linear(16, latent_size, bias=False)
        # one-layer version
        #self.fc1 = nn.Linear(5, 4)
        self.cuda()

    def forward(self, z, actions):
        x = torch.cat([z, actions], dim=1)

        x = self.fc1(x)
        x = F.leaky_relu(x, 0.2)
        x = self.fc2(x)
        x = F.tanh(x)
        return x

# This function identifies the non-pruned connections
def compute_causal_graph(model, latent_size, num_actions):
    input_size = latent_size + num_actions
    rows = []
    for i in range(input_size):
        x = torch.zeros(input_size)
        W1 = model.fc1.weight.abs().cpu()
        W2 = model.fc2.weight.abs().cpu()

        # Continuous version
        zero_return = torch.matmul(W2, torch.matmul(W1, x))# + model.fc2.bias
        x[i] = 1.
        one_return = torch.matmul(W2, torch.matmul(W1, x))# + model.fc2.bias
        result = (one_return - zero_return)

        # Thresholded version
        def binarize(W, theta=.01):
            return (W.abs() > theta).type(torch.FloatTensor)
        #result = torch.matmul(binarize(model.fc2.weight), torch.matmul(binarize(model.fc1.weight), x))
        rows.append(np.array(result.abs().cpu().data))

    scm = np.array(rows)
    #scm -= scm.min()
    eps = .0001
    return scm / (scm.max() + eps)
    #return scm


# Just a helper function to render the graph to an image
def render_causal_graph(scm):
    # Headless matplotlib fix
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    import networkx as nx

    plt.cla()

    # The SCM will have more rows than columns
    # Pad with zeros to create a square adjacency matrix
    rows, cols = scm.shape
    adjacency = np.zeros((rows, rows))
    adjacency[:,:cols] = scm[:]
    print(adjacency)

    edge_alphas = adjacency.flatten() **2

    from networkx.classes.multidigraph import DiGraph
    G = DiGraph(np.ones(adjacency.shape))

    pos = nx.layout.circular_layout(G)

    node_sizes = [10 for i in range(len(G))]
    M = G.number_of_edges()
    #edge_colors = range(2, M + 2)
    #edge_alphas = [(5 + i) / (M + 4) for i in range(M)]
    #edge_colors = [2 for i in range(len(G))]

    nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='blue')
    edges = nx.draw_networkx_edges(G, pos, node_size=node_sizes, arrowstyle='->', arrowsize=20, edge_cmap=plt.cm.Blues, width=2)
    labels = ['$z_{}$'.format(i) for i in range(cols)] + ['$a_{}$'.format(i) for i in range(rows - cols)]
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


def demo_latent_dimensions(before, encoder, decoder, transition, latent_size, num_actions):
    batch_size = before.shape[0]
    actions = torch.zeros(batch_size, num_actions).cuda()
    actions[:, 0] = 1.
    z = transition(encoder(before), actions)
    for i in range(latent_size):
        images = []
        dim_min = z.min(dim=1)[0][i]
        dim_max = z.max(dim=1)[0][i]
        N = batch_size
        for j in range(N):
            dim_range = dim_max - dim_min
            val = dim_min + dim_range * 1.0 * j / N
            zp = z.clone()
            zp[:, i] = val
            images.append(decoder(zp))
        imutil.show(torch.cat(images, dim=0))


def demo_latent_video(before, encoder, decoder, transition, latent_size, num_actions, epoch=0):
    start_time = time.time()
    batch_size = before.shape[0]
    actions = torch.zeros(batch_size, num_actions).cuda()
    actions[:, 0] = 1.  # TODO
    z = transition(encoder(before), actions)
    for i in range(latent_size):
        vid_filename = 'iter_{:06d}_dim_{:02d}'.format(epoch, i)
        vid = imutil.VideoLoop(vid_filename)
        dim_min = z.min(dim=1)[0][i]
        dim_max = z.max(dim=1)[0][i]
        N = 60
        for j in range(N):
            dim_range = dim_max - dim_min
            val = dim_min + dim_range * 1.0 * j / N
            zp = z.clone()
            zp[:, i] = val
            vid.write_frame(decoder(zp), resize_to=(256,256))
        vid.finish()
    print('Finished generating videos in {:03f}s'.format(time.time() - start_time))


# ok now, can the network learn the task?
latent_size = 3
num_actions = 4
data = build_dataset(num_actions)
encoder = Encoder(latent_size)
decoder = Decoder(latent_size)
transition = Transition(latent_size, num_actions)
opt_encoder = optim.Adam(encoder.parameters(), lr=0.0001)
opt_decoder = optim.Adam(decoder.parameters(), lr=0.0001)
opt_transition = optim.Adam(transition.parameters(), lr=0.0001)

iters = 100 * 1000
ts = TimeSeries('Training', iters)

vid = imutil.VideoMaker('causal_model.mp4')
for i in range(iters):
    opt_encoder.zero_grad()
    opt_decoder.zero_grad()
    opt_transition.zero_grad()

    before, actions, target = get_batch(data)

    # Just try to autoencode
    z = encoder(before)
    z_prime = transition(z, actions)
    predicted = decoder(z_prime)

    pred_loss = F.binary_cross_entropy(predicted, target)
    #pred_loss = torch.mean((predicted - target) ** 2)
    ts.collect('Reconstruction loss', pred_loss)

    l1_scale = (5.0 * i) / iters
    l1_loss = 0.
    l1_loss += l1_scale * F.l1_loss(transition.fc1.weight, torch.zeros(transition.fc1.weight.shape).cuda())
    l1_loss += l1_scale * F.l1_loss(transition.fc2.weight, torch.zeros(transition.fc2.weight.shape).cuda())
    ts.collect('Sparsity loss', l1_loss)

    loss = pred_loss + l1_loss

    loss.backward()
    opt_encoder.step()
    opt_decoder.step()
    opt_transition.step()

    if i % 1000 == 0:
        imutil.show(target, caption='Target')
        imutil.show(predicted, caption='Predicted')
        scm = compute_causal_graph(transition, latent_size, num_actions)
        caption = 'Prediction Loss {:.03f}'.format(pred_loss)
        vid.write_frame(render_causal_graph(scm), caption=caption)
        print(transition.fc1.weight)
        print(transition.fc2.weight)
        print(transition.fc2.bias)
        demo_latent_dimensions(before[:9], encoder, decoder, transition, latent_size, num_actions)
        demo_latent_video(before[:9], encoder, decoder, transition, latent_size, num_actions, epoch=i)
    ts.print_every(1)


vid.finish()

print(ts)
