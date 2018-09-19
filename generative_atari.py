import time
import os
import json
import numpy as np
import random

from tqdm import tqdm
import imutil
from logutil import TimeSeries

from atari import MultiEnvironment
from skimage.measure import block_reduce

env = None
prev_states = None

def convert_pong(img_batch):
    batch_size = len(img_batch)
    cropped = np.array(img_batch)[:,32:-18].mean(-1)
    downsampled = np.array([block_reduce(c, (5,5), np.max) for c in cropped])
    downsampled = downsampled - downsampled.min()
    downsampled /= downsampled.max()
    return torch.Tensor(downsampled).view(batch_size, 1, 32, 32).cuda()

def get_batch(batch_size):
    global env
    global prev_states
    num_actions = 4
    if env is None:
        env = MultiEnvironment('Pong-v0', batch_size)
        actions = np.random.randint(0, num_actions, size=batch_size)
        prev_states, rewards, dones, infos = env.step(actions)
        prev_states = convert_pong(prev_states)

    actions = np.random.randint(0, num_actions, size=batch_size)
    states, rewards, dones, infos = env.step(actions)
    states = convert_pong(states)

    action_tensor = torch.Tensor(batch_size, num_actions).zero_().cuda()
    for i in range(batch_size):
        action_tensor[i][actions[i]] = 1.0
    return prev_states, action_tensor, states


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
        self.fc1 = nn.Linear(64*4*4, 128)
        self.bn5 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, latent_size)
        self.cuda()

    def forward(self, x):
        # Input: batch x 32 x 32
        #x = x.unsqueeze(1)
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
        x = self.bn5(x)

        z = self.fc2(x)
        return z


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
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
        self.fc1 = nn.Linear(64*4*4, 128)
        self.fc2 = nn.Linear(128 * 2, 1)
        self.cuda()

    def forward(self, x):
        # Input: batch x 32 x 32
        #x = x.unsqueeze(1)
        # batch x 1 x 32 x 32
        x = self.conv1(x)
        x = F.leaky_relu(x, 0.2)

        x = self.conv2(x)
        x = F.leaky_relu(x, 0.2)

        x = self.conv3(x)
        x = F.leaky_relu(x, 0.2)

        x = self.conv4(x)
        x = F.leaky_relu(x, 0.2)

        x = x.view(-1, 64*4*4)
        x = self.fc1(x)

        # Minibatch Standard Deviation from Karras et al
        augmented = torch.cat([x, torch.var(x, dim=0).expand(32, 128)], 1)
        scores = self.fc2(augmented)
        return scores


class Decoder(nn.Module):
    def __init__(self, latent_size):
        super().__init__()
        self.latent_size = latent_size
        self.fc1 = nn.Linear(latent_size, 128)
        self.bn0 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 256)

        self.deconv1 = nn.ConvTranspose2d(256, 128, 4, stride=1)
        self.bn1 = nn.BatchNorm2d(128)
        # 128 x 4 x 4
        self.deconv2 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        # 64 x 8 x 8
        self.deconv3 = nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        # 32 x 16 x 16
        self.deconv4 = nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        # 16 x 32 x 32
        self.deconv5 = nn.ConvTranspose2d(64, 1, 3, stride=1, padding=1)
        self.cuda()

    def forward(self, z):
        x = self.fc1(z)
        x = F.leaky_relu(x, 0.2)
        x = self.bn0(x)
        x = self.fc2(x)
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
        x = torch.sigmoid(x)
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
        x = torch.tanh(x)
        # Fern hack: Predict a delta/displacement
        return z + x


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

        """
        # Thresholded version
        def binarize(W, theta=.01):
            return (W.abs() > theta).type(torch.FloatTensor)
        result = torch.matmul(binarize(model.fc2.weight), torch.matmul(binarize(model.fc1.weight), x))
        """
        rows.append(np.array(result.abs().cpu().data))

    scm = np.array(rows)
    scm -= scm.min()
    eps = .0001
    return scm / (scm.max() + eps)


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


def demo_latent_video(before, encoder, decoder, transition, latent_size, num_actions, epoch=0):
    start_time = time.time()
    batch_size = before.shape[0]
    actions = torch.zeros(batch_size, num_actions).cuda()
    for i in range(batch_size):
        a = np.random.randint(0, num_actions)
        actions[i, a] = 1

    prev_z = encoder(before)
    z = transition(prev_z, actions)
    for i in range(latent_size):
        vid_filename = 'iter_{:06d}_dim_{:02d}'.format(epoch, i)
        vid = imutil.VideoLoop(vid_filename)
        dim_min = z.min(dim=1)[0][i] - 1.
        dim_max = z.max(dim=1)[0][i] + 1.
        N = 60
        for j in range(N):
            dim_range = dim_max - dim_min
            val = dim_min + dim_range * 1.0 * j / N
            zp = z.clone()
            zp[:, i] = val
            caption = "z{}={:.3f}".format(i, val)
            vid.write_frame(decoder(zp), caption=caption)
        vid.finish()
    print('Finished generating videos in {:03f}s'.format(time.time() - start_time))


def clip_gradients(network, val):
    for W in network.parameters():
        if W.grad is not None:
            W.grad.clamp_(-val, val)


def main():
    # ok now, can the network learn the task?
    latent_size = 4
    num_actions = 4
    batch_size = 32
    encoder = Encoder(latent_size)
    decoder = Decoder(latent_size)
    discriminator = Discriminator()
    transition = Transition(latent_size, num_actions)
    opt_encoder = optim.Adam(encoder.parameters(), lr=0.001)
    opt_decoder = optim.Adam(decoder.parameters(), lr=0.001)
    opt_transition = optim.Adam(transition.parameters(), lr=0.001)
    opt_discriminator = optim.Adam(discriminator.parameters(), lr=0.01)

    iters = 30 * 1000
    ts = TimeSeries('Training', iters)

    vid = imutil.VideoMaker('causal_model.mp4')
    for i in range(iters):

        # First train the discriminator
        for j in range(3):
            opt_discriminator.zero_grad()
            _, _, real = get_batch(batch_size)
            random_z = torch.zeros(batch_size, latent_size).normal_(1, 1).cuda()
            fake = decoder(random_z)
            disc_real = torch.relu(1 + discriminator(real)).sum()
            disc_fake = torch.relu(1 - discriminator(fake)).sum()
            disc_loss = disc_real + disc_fake
            disc_loss.backward()
            clip_gradients(discriminator, 1)
            opt_discriminator.step()
        pixel_variance = fake.var(0).mean()
        ts.collect('Disc real loss', disc_real)
        ts.collect('Disc fake loss', disc_fake)
        ts.collect('Discriminator loss', disc_loss)
        ts.collect('Generated pixel variance', pixel_variance)

        # Apply discriminator loss for realism
        opt_decoder.zero_grad()
        random_z = torch.zeros(batch_size, latent_size).normal_(1, 1).cuda()
        fake = decoder(random_z)
        disc_loss = .01 * torch.relu(1 + discriminator(fake)).sum()
        ts.collect('Gen. Disc loss', disc_loss)
        disc_loss.backward()
        clip_gradients(decoder, 1)
        opt_decoder.step()


        # Now train the autoencoder
        opt_encoder.zero_grad()
        opt_decoder.zero_grad()
        opt_transition.zero_grad()

        before, actions, target = get_batch(batch_size)

        # Just try to autoencode
        z = encoder(before)

        z_prime = transition(z, actions)
        predicted = decoder(z_prime)

        #pred_loss = F.binary_cross_entropy(predicted, target)
        pred_loss = torch.mean((predicted - target) ** 2)
        ts.collect('Reconstruction loss', pred_loss)

        l1_scale = (10.0 * i) / iters
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
            filename = 'iter_{:06}_reconstruction.jpg'.format(i)
            img = torch.cat([target, predicted])
            imutil.show(img, filename=filename)

            scm = compute_causal_graph(transition, latent_size, num_actions)
            caption = 'Prediction Loss {:.03f}'.format(pred_loss)
            vid.write_frame(render_causal_graph(scm), caption=caption)

            demo_latent_video(before[:9], encoder, decoder, transition, latent_size, num_actions, epoch=i)
        ts.print_every(1)


    vid.finish()
    print(ts)


if __name__ == '__main__':
    main()
