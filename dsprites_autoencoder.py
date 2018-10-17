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
        self.fc1 = nn.Linear(64*64, 1200)
        self.bn1 = nn.BatchNorm1d(1200)
        self.fc2 = nn.Linear(1200, 1200)
        self.bn2 = nn.BatchNorm1d(1200)
        self.fc3 = nn.Linear(1200, latent_size * 2)

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
        self.fc1 = nn.Linear(latent_size, 1200)
        self.bn1 = nn.BatchNorm1d(1200)
        self.fc2 = nn.Linear(1200, 1200)
        self.bn2 = nn.BatchNorm1d(1200)
        self.fc3 = nn.Linear(1200, 1200)
        self.bn3 = nn.BatchNorm1d(1200)
        self.fc4 = nn.Linear(1200, 4096)
        # B x 1 x 64 x 64
        self.cuda()

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x, 0.2)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x, 0.2)
        x = self.fc3(x)
        x = self.bn3(x)
        x = F.leaky_relu(x, 0.2)
        x = self.fc4(x)
        #x = torch.sigmoid(x)
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
    opt_enc = torch.optim.Adagrad(encoder.parameters(), lr=.01)
    opt_dec = torch.optim.Adagrad(decoder.parameters(), lr=.01)
    train_iters = 25 * 1000
    ts = TimeSeries('Training Autoencoder', train_iters)
    for train_iter in range(train_iters + 1):
        encoder.train()
        decoder.train()

        random_factors = np.random.uniform(size=(batch_size, true_latent_dim))
        images, factors = dsprites.get_batch()
        opt_enc.zero_grad()
        opt_dec.zero_grad()
        x = torch.Tensor(images).cuda()
        mu, log_variance = encoder(x)
        reconstructed_logits = decoder(reparameterize(mu, log_variance))
        reconstructed = torch.sigmoid(reconstructed)

        #mse_loss = torch.mean((x - reconstructed) ** 2)
        recon_loss = F.binary_cross_entropy_with_logits(reconstructed_logits, x)
        ts.collect('Reconstruction loss', recon_loss)

        kld_loss = -0.5 * torch.mean(1 + log_variance - mu.pow(2) - log_variance.exp())
        ts.collect('KLD loss', kld_loss)

        loss = recon_loss + kld_loss
        loss.backward()
        opt_enc.step()
        opt_dec.step()
        ts.print_every(2)

        encoder.eval()
        decoder.eval()

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
