import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import imutil
from logutil import TimeSeries
from tqdm import tqdm

import dsprites
from higgins import higgins_metric


class Encoder(nn.Module):
    def __init__(self, latent_size):
        super().__init__()
        self.latent_size = latent_size
        # Bx1x64x64
        self.conv1 = nn.Conv2d(1, 32, 4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        # Bx1x32x32
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        # Bx32x16x16
        self.conv3 = nn.Conv2d(64, 64, 4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        # Bx64x8x8
        self.conv4 = nn.Conv2d(64, 64, 4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        # Bx64x4x4
        #self.conv5 = nn.Conv2d(64, 64, 4, stride=2, padding=1)
        #self.bn5 = nn.BatchNorm2d(64)
        # Bx64x2x2
        self.fc1 = nn.Linear(64*2*2, 256)
        self.bn_fc1 = nn.BatchNorm1d(256)
        # Bx256
        self.fc2 = nn.Linear(256, latent_size)
        # Bxlatent_size
        self.cuda()

    def forward(self, x):
        # Input: B x 1 x 64 x 64
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

        #x = self.conv5(x)
        #x = self.bn5(x)
        #x = F.leaky_relu(x, 0.2)

        x = x.view(-1, 64*2*2)
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = F.leaky_relu(x, 0.2)

        z = self.fc2(x)
        return z


class Decoder(nn.Module):
    def __init__(self, latent_size):
        super().__init__()
        self.latent_size = latent_size
        self.fc1 = nn.Linear(latent_size, 256)

        self.deconv1 = nn.ConvTranspose2d(256, 128, 4, stride=1)
        self.bn1 = nn.BatchNorm2d(128)
        # B x 128 x 4 x 4
        self.deconv2 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        # B x 64 x 8 x 8
        self.deconv3 = nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        # B x 64 x 16 x 16
        self.deconv4 = nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        # B x 64 x 32 x 32
        self.deconv5 = nn.ConvTranspose2d(64, 1, 3, stride=1, padding=1)
        # B x 1 x 32 x 32
        self.cuda()

    def forward(self, z):
        x = self.fc1(z)
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

def main():
    dsprites.init()

    # Compute Higgins metric for a randomly-initialized convolutional encoder
    batch_size = 32
    latent_dim = 10
    true_latent_dim = 4
    encoder = Encoder(latent_dim)
    decoder = Decoder(latent_dim)
    random_score = higgins_metric(dsprites.simulator, true_latent_dim, encoder, latent_dim)

    # Train the autoencoder
    opt_enc = torch.optim.Adam(encoder.parameters())
    opt_dec = torch.optim.Adam(decoder.parameters())
    train_iters = 10 * 1000
    ts = TimeSeries('Training Autoencoder', train_iters)
    for train_iter in range(train_iters):
        random_factors = np.random.uniform(size=(batch_size, true_latent_dim))
        images, factors = dsprites.get_batch()
        opt_enc.zero_grad()
        opt_dec.zero_grad()
        x = torch.Tensor(images).cuda()
        z = encoder(x)
        reconstructed = decoder(z)
        loss = torch.sum((x - reconstructed) ** 2)
        loss.backward()
        ts.collect('MSE loss', loss)
        opt_enc.step()
        opt_dec.step()
        ts.print_every(2)

        if train_iter % 1000 == 0:
            filename = 'vis_iter_{:06d}.jpg'.format(train_iter)
            imutil.show(torch.cat([x[:4], reconstructed[:4]], dim=0), caption='Reconstruction iter {}'.format(train_iter), filename=filename, font_size=12)
            # Compute metric again after training
            trained_score = higgins_metric(dsprites.simulator, true_latent_dim, encoder, latent_dim)

            print('Higgins metric before training: {}'.format(random_score))
            print('Higgins metric after training {} iters: {}'.format(
                train_iter, trained_score))


if __name__ == '__main__':
    main()
