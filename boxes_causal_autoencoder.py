import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import imutil
from logutil import TimeSeries
from tqdm import tqdm

import boxes
from higgins import higgins_metric


class Transition(nn.Module):
    def __init__(self, latent_size, num_actions):
        super().__init__()
        # Input: State + Action
        # Output: State
        self.fc1 = nn.Linear(latent_size + num_actions, 16, bias=False)
        self.fc2 = nn.Linear(16, latent_size, bias=False)
        self.cuda()

    def forward(self, z, actions):
        x = torch.cat([z, actions], dim=1)
        x = self.fc1(x)
        x = F.leaky_relu(x, 0.2)
        x = self.fc2(x)
        x = torch.tanh(x)
        # Fern hack: Predict a delta/displacement
        return z + x


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
        x = F.relu(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.fc3(x)
        return x, None


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


def main():
    boxes.init()

    # Compute Higgins metric for a randomly-initialized convolutional encoder
    batch_size = 32
    latent_dim = 4
    true_latent_dim = 4
    num_actions = 4
    encoder = Encoder(latent_dim)
    decoder = Decoder(latent_dim)
    transition = Transition(latent_dim, num_actions)
    higgins_scores = []

    # Train the autoencoder
    opt_enc = torch.optim.Adagrad(encoder.parameters(), lr=.01)
    opt_dec = torch.optim.Adagrad(decoder.parameters(), lr=.01)
    opt_trans = torch.optim.Adagrad(transition.parameters(), lr=.01)
    train_iters = 50 * 1000
    ts = TimeSeries('Training Autoencoder', train_iters)
    for train_iter in range(train_iters + 1):
        encoder.train()
        decoder.train()

        images, actions, images_tplusone = boxes.get_batch()
        opt_enc.zero_grad()
        opt_dec.zero_grad()
        x = torch.Tensor(images).cuda().unsqueeze(1)
        a_t = torch.Tensor(actions).cuda()
        x_tplusone = torch.Tensor(images_tplusone).cuda().unsqueeze(1)
        z, _ = encoder(x)
        reconstructed_logits = decoder(z)
        reconstructed = torch.sigmoid(reconstructed_logits)

        #recon_loss = torch.sum((x - reconstructed) ** 2)
        #recon_loss = F.binary_cross_entropy_with_logits(reconstructed_logits, x, reduction='sum')
        #ts.collect('Reconstruction loss', recon_loss)

        z_tplusone = transition(z, a_t)
        predicted_logits = decoder(z_tplusone)
        prediction_loss = F.binary_cross_entropy_with_logits(predicted_logits, x_tplusone, reduction='sum')
        ts.collect('Prediction loss', prediction_loss)

        l1_scale = 10.
        #l1_loss = 0.
        #l1_loss += l1_scale * F.l1_loss(transition.fc1.weight, torch.zeros(transition.fc1.weight.shape).cuda())
        #l1_loss += l1_scale * F.l1_loss(transition.fc2.weight, torch.zeros(transition.fc2.weight.shape).cuda())
        z_diff = z_tplusone - z
        l1_loss = l1_scale * F.l1_loss(z_diff, torch.zeros(z_diff.shape).cuda())
        ts.collect('Sparsity loss', l1_loss)

        loss = prediction_loss + l1_loss
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
            imutil.show(img, filename=filename, caption='D(E(x)) iter {}'.format(train_iter), font_size=10)
        if train_iter % 10000 == 0:
            # Compute metric again after training
            trained_score = higgins_metric(boxes.simulator, true_latent_dim, encoder, latent_dim)
            higgins_scores.append(trained_score)
            print('Higgins metric before training: {}'.format(higgins_scores[0]))
            print('Higgins metric after training {} iters: {}'.format(train_iter, higgins_scores[-1]))
            print('Best Higgins: {}'.format(max(higgins_scores)))
            ts.collect('Higgins Metric', trained_score)
    print(ts)
    print('Finished')


if __name__ == '__main__':
    main()
