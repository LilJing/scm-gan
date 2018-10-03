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

def main():
    dsprites.init()

    # Compute Higgins metric for a randomly-initialized convolutional encoder
    latent_size = 4
    encoder = Encoder(latent_size)
    score = higgins_metric(dsprites.simulator, 4, encoder, latent_size)

if __name__ == '__main__':
    main()
