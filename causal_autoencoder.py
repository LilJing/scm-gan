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
        x = norm(x)
        return x


class Encoder(nn.Module):
    def __init__(self, latent_size):
        super().__init__()
        self.latent_size = latent_size
        # Bx1x64x64
        self.conv1 = nn.Conv2d(1, 8, 4, stride=2, padding=1)
        self.bn_conv1 = nn.BatchNorm2d(8)
        # Bx8x32x32
        self.conv2 = nn.Conv2d(8, 32, 4, stride=2, padding=1)
        self.bn_conv2 = nn.BatchNorm2d(32)

        self.fc1 = nn.Linear(32*16*16, 196)
        self.bn1 = nn.BatchNorm1d(196)
        self.fc2 = nn.Linear(196, 196)
        self.bn2 = nn.BatchNorm1d(196)
        self.fc3 = nn.Linear(196, latent_size)

        # Bxlatent_size
        self.cuda()

    def forward(self, x):
        # Input: B x 1 x 64 x 64
        x = self.conv1(x)
        x = self.bn_conv1(x)
        x = F.leaky_relu(x)

        x = self.conv2(x)
        x = self.bn_conv2(x)
        x = F.leaky_relu(x)

        x = x.view(-1, 32*16*16)
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x)

        x = self.fc3(x)
        x = norm(x)
        return x


class Decoder(nn.Module):
    def __init__(self, latent_size, k=64):
        super().__init__()
        self.latent_size = latent_size
        self.k = k
        self.fc1 = nn.Linear(latent_size*k, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 196)
        self.bn2 = nn.BatchNorm1d(196)
        self.fc3 = nn.Linear(196, 2048)
        self.bn3 = nn.BatchNorm1d(2048)

        self.conv1 = nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1)
        self.bn_conv1 = nn.BatchNorm2d(32)
        self.conv2 = nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1)
        self.bn_conv2 = nn.BatchNorm2d(32)
        self.conv3 = nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1)

        self.sod1 = SelfOrganizingDiscretization(z=latent_size, k=k)
        self.cuda()



    def forward(self, x):
        x = self.sod1(x)
        x = x.reshape(-1, self.latent_size*self.k)

        x = self.fc1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x, 0.2)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x, 0.2)
        x = self.fc3(x)
        x = self.bn3(x)
        x = F.leaky_relu(x, 0.2)

        x = x.view(-1, 32, 8, 8)
        x = self.conv1(x)
        x = self.bn_conv1(x)
        x = F.leaky_relu(x, 0.2)
        x = self.conv2(x)
        x = self.bn_conv2(x)
        x = F.leaky_relu(x, 0.2)
        x = self.conv3(x)
        x = F.leaky_relu(x, 0.2)
        return x


class SelfOrganizingDiscretization(nn.Module):
    def __init__(self, z, k, kernel='inverse_multiquadratic'):
        super().__init__()
        self.z = z
        self.k = k
        rho = torch.arange(-1, 1, 2/k).unsqueeze(0).repeat(z, 1)
        self.particles = torch.nn.Parameter(rho)
        self.cuda()

    def forward(self, x):
        # x is a real-valued tensor size (batch, Z)
        batch_size = len(x)
        # Broadcast x to (batch, Z, K)
        perceived_locations = x.unsqueeze(-1).repeat(1, 1, self.k)
        reference_locations = self.particles.unsqueeze(0).repeat(batch_size, 1, 1)
        distances = (perceived_locations - reference_locations) ** 2
        # IMQ kernel
        kern = .01 / (.01 + distances)
        # Gaussian RBF kernel
        # kern = torch.exp(-distances)
        # Output is a category between 1 and K, for each of the Z real values
        return torch.softmax(kern, dim=2)




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
    z_fake = norm(z_fake)
    mmd_loss = -imq_kernel(z, z_fake, h_dim=latent_dim)
    return mmd_loss.mean()


# Normalize a batch of latent points to the unit hypersphere
def norm(x):
    norm = torch.norm(x, p=2, dim=1)
    x = x / (norm.expand(1, -1).t() + .0001)
    return x


def main():
    datasource.init()

    # Compute Higgins metric for a randomly-initialized convolutional encoder
    batch_size = 64
    latent_dim = 4
    true_latent_dim = 4
    num_actions = 4
    encoder = Encoder(latent_dim)
    decoder = Decoder(latent_dim)
    transition = Transition(latent_dim, num_actions)
    higgins_scores = []

    # Train the autoencoder
    opt_enc = torch.optim.Adam(encoder.parameters(), lr=.001)
    opt_dec = torch.optim.Adam(decoder.parameters(), lr=.001)
    opt_trans = torch.optim.Adam(transition.parameters(), lr=.001)
    train_iters = 50 * 1000
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
        recon_loss = F.binary_cross_entropy_with_logits(reconstructed_logits, x)
        ts.collect('Reconstruction loss', recon_loss)

        z_tplusone = transition(z, a_t)
        predicted_logits = decoder(z_tplusone)
        predicted = torch.sigmoid(predicted_logits)
        #prediction_loss = torch.sum((predicted - x_tplusone) ** 2)
        prediction_loss = F.binary_cross_entropy_with_logits(predicted_logits, x_tplusone)
        ts.collect('Prediction loss', prediction_loss)

        # MMD penalty for the future
        mmd_loss = 0
        z_t = z
        for i in range(1):
            z_t = transition(z_t, a_t)
            mmd_loss += 1000 * mmd_normal_penalty(z_t)
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
            for frame in range(60):
                z = transition(z, a_t)
                a_t = torch.cat((a_t[-1:], a_t[:-1]))
                predicted = torch.sigmoid(decoder(z))
                img = torch.cat((x[:4], predicted[:4]), dim=3)
                caption = 'Pred. t+{} a={}'.format(frame, torch.argmax(a_t[:4], dim=1).cpu().numpy())
                vid.write_frame(img, filename=filename, caption=caption, img_padding=8, font_size=10, resize_to=(800,400))
                vid.write_frame(img, filename=filename, caption=caption, img_padding=8, font_size=10, resize_to=(800,400))
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
