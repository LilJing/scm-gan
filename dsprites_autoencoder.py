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
        self.fc1 = nn.Linear(64*64, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.fc3 = nn.Linear(1024, latent_size * 2)

        # Bxlatent_size
        self.cuda()

    def forward(self, x):
        # Input: B x 1 x 64 x 64
        x = x.view(-1, 64*64)
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.fc3(x)
        return x[:,:self.latent_size], x[:, self.latent_size:]


class Decoder(nn.Module):
    def __init__(self, latent_size):
        super().__init__()
        self.latent_size = latent_size
        self.fc1 = nn.Linear(latent_size, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 4096)
        # B x 1 x 64 x 64
        self.cuda()

    def forward(self, x):
        x = self.fc1(x)
        x = F.leaky_relu(x, 0.2)
        x = self.fc2(x)
        x = F.leaky_relu(x, 0.2)
        x = self.fc3(x)
        x = torch.sigmoid(x)
        x = x.view(-1, 1, 64, 64)
        return x


def reparameterize(mu, logvar):
    std = torch.exp(0.5*logvar)
    eps = torch.randn_like(std)
    return eps.mul(std).add_(mu)


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
    train_iters = 20 * 1000
    ts = TimeSeries('Training Autoencoder', train_iters)
    for train_iter in range(train_iters + 1):
        random_factors = np.random.uniform(size=(batch_size, true_latent_dim))
        images, factors = dsprites.get_batch()
        opt_enc.zero_grad()
        opt_dec.zero_grad()
        x = torch.Tensor(images).cuda()
        mu, sigma = encoder(x)
        reconstructed = decoder(reparameterize(mu, sigma))
        loss = torch.sum((x - reconstructed) ** 2)
        loss.backward()
        ts.collect('MSE loss', loss)
        opt_enc.step()
        opt_dec.step()
        ts.print_every(2)

        if train_iter % 1000 == 0:
            filename = 'vis_iter_{:06d}.jpg'.format(train_iter)
            img = torch.cat((x[:4], reconstructed[:4]), dim=3)
            imutil.show(img, filename=filename, caption='D(E(x)) iter {}'.format(train_iter), font_size=10)
            # Compute metric again after training

            trained_score = higgins_metric(dsprites.simulator, true_latent_dim, encoder, latent_dim)
            print('Higgins metric before training: {}'.format(random_score))
            print('Higgins metric after training {} iters: {}'.format(
                train_iter, trained_score))
            ts.collect('Higgins Metric', trained_score)
    print(ts)
    print('Finished')


if __name__ == '__main__':
    main()
