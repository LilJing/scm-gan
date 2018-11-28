import sys
if len(sys.argv) < 2:
    print('Usage: {} datasource'.format(sys.argv[0]))
    print('\tdatasource: boxes/minipong/...')
    exit(1)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import imutil
from logutil import TimeSeries
from tqdm import tqdm

from higgins import higgins_metric

from importlib import import_module
datasource = import_module(sys.argv[1])


class Transition(nn.Module):
    def __init__(self, latent_size, num_actions):
        super().__init__()
        # Input: State + Action
        # Output: State
        self.fc1 = nn.Linear(latent_size + num_actions, 16)
        self.fc2 = nn.Linear(16, latent_size)
        self.cuda()

    def forward(self, z, actions):
        x = torch.cat([z, actions], dim=1)
        x = self.fc1(x)
        x = F.leaky_relu(x, 0.2)
        x = self.fc2(x)

        # Fern hack: Predict a delta/displacement
        x = x + z

        # Normalize to the ball
        norm = torch.norm(x, p=2, dim=1)
        x = x / (norm.expand(1, -1).t() + .0001)
        return x


class Encoder(nn.Module):
    def __init__(self, latent_size):
        super().__init__()
        self.latent_size = latent_size
        # Bx1x64x64
        self.fc1 = nn.Linear(64*64, 120)
        self.bn1 = nn.BatchNorm1d(120)
        self.fc2 = nn.Linear(120, 120)
        self.bn2 = nn.BatchNorm1d(120)
        self.fc3 = nn.Linear(120, latent_size)

        # Bxlatent_size
        self.cuda()

    def forward(self, x):
        # Input: B x 1 x 64 x 64
        x = x.view(-1, 64*64)
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x)

        x = self.fc3(x)
        # Normalize to the ball
        norm = torch.norm(x, p=2, dim=1)
        x = x / (norm.expand(1, -1).t() + .0001)
        return x


class Decoder(nn.Module):
    def __init__(self, latent_size):
        super().__init__()
        self.latent_size = latent_size
        self.fc1 = nn.Linear(latent_size, 120)
        self.bn1 = nn.BatchNorm1d(120)
        self.fc2 = nn.Linear(120, 120)
        self.bn2 = nn.BatchNorm1d(120)
        self.fc3 = nn.Linear(120, 120)
        self.bn3 = nn.BatchNorm1d(120)
        self.fc4 = nn.Linear(120, 4096)
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


# Inverse multiquadratic kernel with varying kernel bandwidth
# Tolstikhin et al. https://arxiv.org/abs/1711.01558
# https://github.com/schelotto/Wasserstein_Autoencoders
def imq_kernel(X: torch.Tensor,
               Y: torch.Tensor,
               h_dim: int):
    batch_size = X.size(0)

    p2_norm_x = X.pow(2).sum(1).unsqueeze(0)
    norms_x = X.sum(1).unsqueeze(0)
    prods_x = torch.mm(norms_x, norms_x.t())
    dists_x = p2_norm_x + p2_norm_x.t() - 2 * prods_x

    p2_norm_y = Y.pow(2).sum(1).unsqueeze(0)
    norms_y = X.sum(1).unsqueeze(0)
    prods_y = torch.mm(norms_y, norms_y.t())
    dists_y = p2_norm_y + p2_norm_y.t() - 2 * prods_y

    dot_prd = torch.mm(norms_x, norms_y.t())
    dists_c = p2_norm_x + p2_norm_y.t() - 2 * dot_prd

    stats = 0
    for scale in [.1, .2, .5, 1., 2., 5., 10.]:
        C = 2 * h_dim * 1.0 * scale
        res1 = C / (C + dists_x)
        res1 += C / (C + dists_y)

        if torch.cuda.is_available():
            res1 = (1 - torch.eye(batch_size).cuda()) * res1
        else:
            res1 = (1 - torch.eye(batch_size)) * res1

        res1 = res1.sum() / (batch_size - 1)
        res2 = C / (C + dists_c)
        res2 = res2.sum() * 2. / (batch_size)
        stats += res1 - res2
    return stats


# Maximum Mean Discrepancy between z and a reference distribution
# This term goes to zero if z is perfectly normal (with variance sigma**2)
def mmd_normal_penalty(z, sigma=1.0):
    batch_size, latent_dim = z.shape
    z_fake = torch.randn(batch_size, latent_dim).cuda() * sigma

    # Normalize to the ball
    norm = torch.norm(z_fake, p=2, dim=1)
    z_fake = z_fake / (norm.expand(1, -1).t() + .0001)


    mmd_loss = -imq_kernel(z, z_fake, h_dim=latent_dim)
    return mmd_loss.mean()


def main():
    datasource.init()

    # Compute Higgins metric for a randomly-initialized convolutional encoder
    batch_size = 32
    latent_dim = 6
    true_latent_dim = 6
    num_actions = 4
    encoder = Encoder(latent_dim)
    decoder = Decoder(latent_dim)
    transition = Transition(latent_dim, num_actions)
    higgins_scores = []

    # Train the autoencoder
    opt_enc = torch.optim.Adam(encoder.parameters(), lr=.001)
    opt_dec = torch.optim.Adam(decoder.parameters(), lr=.001)
    opt_trans = torch.optim.Adam(transition.parameters(), lr=.001)
    train_iters = 100 * 1000
    ts = TimeSeries('Training Autoencoder', train_iters)
    for train_iter in range(train_iters + 1):
        encoder.train()
        decoder.train()
        transition.train()

        opt_enc.zero_grad()
        opt_dec.zero_grad()
        opt_trans.zero_grad()

        images, actions, images_tplusone = datasource.get_batch()
        x = torch.Tensor(images).cuda().unsqueeze(1)
        a_t = torch.Tensor(actions).cuda()
        x_tplusone = torch.Tensor(images_tplusone).cuda().unsqueeze(1)
        z = encoder(x)
        reconstructed_logits = decoder(z)
        reconstructed = torch.sigmoid(reconstructed_logits)

        #recon_loss = torch.sum((x - reconstructed) ** 2)
        recon_loss = F.binary_cross_entropy_with_logits(reconstructed_logits, x, reduction='sum')
        ts.collect('Reconstruction loss', recon_loss)

        z_tplusone = transition(z, a_t)
        predicted_logits = decoder(z_tplusone)
        predicted = torch.sigmoid(predicted_logits)
        #prediction_loss = torch.sum((predicted - x_tplusone) ** 2)
        prediction_loss = F.binary_cross_entropy_with_logits(predicted_logits, x_tplusone, reduction='sum')
        ts.collect('Prediction loss', prediction_loss)

        mmd_loss = 1000 * (mmd_normal_penalty(z) + mmd_normal_penalty(z_tplusone))
        ts.collect('MMD Loss', mmd_loss)

        loss = prediction_loss + recon_loss + mmd_loss
        loss.backward()
        opt_enc.step()
        opt_dec.step()
        opt_trans.step()
        ts.print_every(2)

        encoder.eval()
        decoder.eval()
        transition.eval()

        if train_iter % 1000 == 0:
            filename = 'vis_iter_{:06d}.jpg'.format(train_iter)
            img = torch.cat((x[:4], reconstructed[:4]), dim=3)
            caption = 'D(E(x)) iter {}'.format(train_iter)
            imutil.show(img, filename=filename, caption=caption, img_padding=4, font_size=10)
        if train_iter % 2000 == 0:
            vid = imutil.VideoMaker('simulation_iter_{:06d}.jpg'.format(train_iter))
            for frame in range(100):
                z = transition(z, a_t)
                predicted = torch.sigmoid(decoder(z))
                img = torch.cat((x[:4], predicted[:4]), dim=3)
                caption = 'Est. future t+{}'.format(frame)
                vid.write_frame(img, filename=filename, caption=caption, img_padding=4, font_size=10)
            vid.finish()
        """"
        if train_iter % 10000 == 0:
            # Compute metric again after training
            trained_score = higgins_metric(datasource.simulator, true_latent_dim, encoder, latent_dim)
            higgins_scores.append(trained_score)
            print('Higgins metric before training: {}'.format(higgins_scores[0]))
            print('Higgins metric after training {} iters: {}'.format(train_iter, higgins_scores[-1]))
            print('Best Higgins: {}'.format(max(higgins_scores)))
            ts.collect('Higgins Metric', trained_score)
        """
    print(ts)
    print('Finished')


if __name__ == '__main__':
    main()
