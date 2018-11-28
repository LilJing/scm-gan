import time
import os
import json
import numpy as np
import random

from tqdm import tqdm
import imutil
from logutil import TimeSeries
from causal_graph import compute_causal_graph, render_causal_graph
import minipong

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


latent_size = 4
l1_power = 10.0
disc_power = .001
num_actions = 4
batch_size = 32
iters = 25 * 1000


dataset = minipong.build_dataset(num_actions)


class Encoder(nn.Module):
    def __init__(self, latent_size):
        super().__init__()
        self.latent_size = latent_size
        # 1x40x40
        self.conv1 = nn.Conv2d(3, 32, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        # 32x40x40
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        # 64x20x20
        self.conv3 = nn.Conv2d(64, 64, 4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        # 64x10x10
        self.conv4 = nn.Conv2d(64, 64, 4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        # 64x5x5
        self.conv5 = nn.Conv2d(64, 64, 3, stride=1, padding=0)
        self.bn5 = nn.BatchNorm2d(64)
        # 64x3x3
        self.fc1 = nn.Linear(64*3*3, 128)
        self.bn_fc1 = nn.BatchNorm1d(128)
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

        x = self.conv5(x)
        x = self.bn5(x)
        x = F.leaky_relu(x, 0.2)

        x = x.view(-1, 64*3*3)
        x = self.fc1(x)
        x = self.bn_fc1(x)

        z = self.fc2(x)
        return z


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        img_width = 40
        hidden_size = 256
        self.fc1a = nn.Linear(img_width, hidden_size//2)
        self.fc1b = nn.Linear(img_width, hidden_size//2)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.cuda()

    def forward(self, x):
        # A constrained discriminator
        # It can see only projections, 1D shadows of the 2D input

        # Input: batch x 3 x width x width
        x = x.mean(dim=1)
        # batch x width x width
        latitude = self.fc1a(x.mean(dim=1))
        longitude = self.fc1a(x.mean(dim=2))
        # (batch x width) + (batch x width)
        x = torch.cat([latitude, longitude], dim=1)
        # batch x hidden_size
        x = F.leaky_relu(x, 0.2)
        x = self.fc2(x)
        return x


class Decoder(nn.Module):
    def __init__(self, latent_size):
        super().__init__()
        self.latent_size = latent_size
        self.fc1 = nn.Linear(latent_size, 128)
        self.bn0 = nn.BatchNorm1d(128)

        self.fc2 = nn.Linear(128, 256)

        self.deconv1 = nn.ConvTranspose2d(256, 128, 5, stride=1)
        self.bn1 = nn.BatchNorm2d(128)
        # 128 x 5 x 5
        self.deconv2 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        # 64 x 10 x 10
        self.deconv3 = nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        # 64 x 20 x 20
        self.deconv4 = nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        # 64 x 40 x 40
        self.deconv5 = nn.ConvTranspose2d(64, 3, 3, stride=1, padding=1)
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
        self.hidden_size = latent_size * 2
        self.fc1 = nn.Linear(latent_size + num_actions, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size ** 2, latent_size, bias=False)
        self.cuda()

    def forward(self, z, actions):
        batch_size = len(z)
        x = torch.cat([z, actions], dim=1)

        x = self.fc1(x)
        x = torch.sigmoid(x)

        # Xiaoli idea: bilinear layer
        # Multilinear Einstein summation to compute a differentiable OR between all inputs
        x = torch.einsum('bi,bj->bij', (x, x)).view(batch_size, -1)
        x = self.fc2(x)

        # Alan hack: Predict a delta/displacement
        return z + x



def demo_latent_video(before, encoder, decoder, transition, latent_size, num_actions, epoch=0):
    start_time = time.time()
    z = encoder(before)
    for i in range(latent_size):
        vid_filename = 'iter_{:06d}_dim_{:02d}'.format(epoch, i)
        vid = imutil.VideoLoop(vid_filename)
        dim_min = z[:,i].min().item()
        dim_max = z[:,i].max().item()
        N = 60
        for j in range(N):
            dim_range = dim_max - dim_min
            val = dim_min + dim_range * 1.0 * j / N
            zp = z.clone()
            zp[:, i] = val
            caption = "z{}={:.3f}".format(i, val)
            img = decoder(zp)
            vid.write_frame(img, caption=caption, resize_to=(256,256), img_padding=10)
        vid.finish()
    print('Finished generating videos in {:03f}s'.format(time.time() - start_time))


def demo_interpolation_video(before, encoder, decoder, transition, latent_size, num_actions, epoch=0):
    start_time = time.time()

    # Take the first two images and interpolate between them
    z = encoder(before[:2])
    start_z = z[0:1]
    end_z = z[1:2]

    vid_filename = 'iter_{:06d}_interp'.format(epoch)
    vid = imutil.VideoLoop(vid_filename)

    for _ in range(10):
        vid.write_frame(before[0], resize_to=(256,256), caption='Start')

    N = 60
    for j in range(N):
        val = 1.0 * j / N
        zp = val * end_z + (1 - val) * start_z
        img = decoder(zp)
        caption = "Interp {:.3f}".format(val)
        vid.write_frame(img, caption=caption, resize_to=(256,256))

    for _ in range(10):
        vid.write_frame(before[1], resize_to=(256,256), caption='End')
    vid.finish()
    print('Finished generating interpolation in {:03f}s'.format(time.time() - start_time))


def clip_gradients(network, val):
    for W in network.parameters():
        if W.grad is not None:
            W.grad.clamp_(-val, val)


def clip_weights(network, val):
    for W in network.parameters():
        if W.data is not None:
            W.data.clamp_(-val, val)


def main():
    # ok now, can the network learn the task?
    encoder = Encoder(latent_size)
    decoder = Decoder(latent_size)
    discriminator = Discriminator()
    transition = Transition(latent_size, num_actions)
    opt_encoder = optim.Adam(encoder.parameters(), lr=0.0001)
    opt_decoder = optim.Adam(decoder.parameters(), lr=0.0001)
    opt_transition = optim.Adam(transition.parameters(), lr=0.0001)
    opt_discriminator = optim.Adam(discriminator.parameters(), lr=0.0001)

    ts = TimeSeries('Training', iters)

    vid = imutil.VideoMaker('causal_model.mp4')
    for i in range(iters):
        encoder.train()
        decoder.train()
        discriminator.train()
        transition.train()

        # First train the discriminator
        for j in range(3):
            opt_discriminator.zero_grad()
            _, _, real = minipong.get_batch(dataset)
            fake = decoder(encoder(real))
            disc_real = torch.relu(1 + discriminator(real)).sum()
            disc_fake = torch.relu(1 - discriminator(fake)).sum()
            disc_loss = disc_real + disc_fake
            disc_loss.backward()
            #clip_gradients(discriminator, 1)
            opt_discriminator.step()
        pixel_variance = fake.var(0).mean()
        ts.collect('Disc real loss', disc_real)
        ts.collect('Disc fake loss', disc_fake)
        ts.collect('Discriminator loss', disc_loss)
        ts.collect('Generated pixel variance', pixel_variance)
        #clip_weights(discriminator, .01)

        # Apply discriminator loss for realism
        opt_decoder.zero_grad()
        _, _, real = minipong.get_batch(dataset)
        fake = decoder(encoder(real))
        disc_loss = disc_power * torch.relu(1 + discriminator(fake)).sum()
        ts.collect('Gen. Disc loss', disc_loss)
        disc_loss.backward()
        #clip_gradients(decoder, .01)
        opt_decoder.step()

        # Now train the autoencoder
        opt_encoder.zero_grad()
        opt_decoder.zero_grad()
        opt_transition.zero_grad()

        before, actions, target = get_batch(dataset)

        z = encoder(before)
        z_prime = transition(z, actions)
        predicted = decoder(z_prime)

        pred_loss = F.binary_cross_entropy(predicted, target)
        #pred_loss = torch.mean((predicted - target) ** 2)
        ts.collect('Reconstruction loss', pred_loss)

        l1_scale = (l1_power * i) / iters
        l1_loss = 0.
        l1_loss += l1_scale * F.l1_loss(transition.fc1.bias, torch.zeros(transition.fc1.bias.shape).cuda())
        l1_loss += l1_scale * F.l1_loss(transition.fc1.weight, torch.zeros(transition.fc1.weight.shape).cuda())
        l1_loss += l1_scale * F.l1_loss(transition.fc2.weight, torch.zeros(transition.fc2.weight.shape).cuda())
        ts.collect('Sparsity loss', l1_loss)

        loss = pred_loss + l1_loss

        loss.backward()
        opt_encoder.step()
        opt_decoder.step()
        opt_transition.step()

        encoder.eval()
        decoder.eval()
        discriminator.eval()
        transition.eval()

        if i % 100 == 0:
            filename = 'iter_{:06}_reconstruction.jpg'.format(i)
            img = torch.cat([target[:2], predicted[:2]])
            caption = 'iter {}: top orig. bot recon.'.format(i)
            imutil.show(img, filename=filename, resize_to=(256,256), img_padding=10, caption=caption)

            scm = compute_causal_graph(transition, latent_size, num_actions)
            caption = 'Iteration {} Prediction Loss {:.03f}'.format(i, pred_loss)
            vid.write_frame(render_causal_graph(scm), caption=caption)

        if i % 1000 == 0:
            demo_latent_video(before[:9], encoder, decoder, transition, latent_size, num_actions, epoch=i)
            demo_interpolation_video(before[:2], encoder, decoder, transition, latent_size, num_actions, epoch=i)
        ts.print_every(2)


    vid.finish()
    print(ts)


if __name__ == '__main__':
    main()