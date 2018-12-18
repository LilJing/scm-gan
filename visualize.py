import time
import math
import os
import sys
if len(sys.argv) < 2:
    print('Usage: {} datasource'.format(sys.argv[0]))
    print('\tAvailable datasources: boxes, minipong, mediumpong...')
    exit(1)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import imutil
from logutil import TimeSeries
from tqdm import tqdm
from spatial_recurrent import CSRN
from coordconv import CoordConv2d

from higgins import higgins_metric

from importlib import import_module
datasource = import_module('envs.' + sys.argv[1])

import models


def main():
    batch_size = 16
    latent_dim = 12
    true_latent_dim = 4
    num_actions = 4
    encoder = models.Encoder(latent_dim)
    decoder = models.Decoder(latent_dim)
    transition = models.Transition(latent_dim, num_actions)
    blur = models.GaussianSmoothing(channels=3, kernel_size=11, sigma=4.)
    higgins_scores = []

    #load_from_dir = '/mnt/nfs/experiments/demo_2018_12_12/scm-gan_81bd12cd'
    load_from_dir = '.'

    print('Loading models from directory {}'.format(load_from_dir))
    encoder.load_state_dict(torch.load(os.path.join(load_from_dir, 'model-encoder.pth')))
    decoder.load_state_dict(torch.load(os.path.join(load_from_dir, 'model-decoder.pth')))
    transition.load_state_dict(torch.load(os.path.join(load_from_dir, 'model-transition.pth')))

    encoder.eval()
    decoder.eval()
    transition.eval()
    for model in (encoder, decoder, transition):
        for child in model.children():
            if type(child) == nn.BatchNorm2d or type(child) == nn.BatchNorm1d:
                child.momentum = 0

    states, rewards, dones, actions = datasource.get_trajectories(batch_size, timesteps=1)
    states = torch.Tensor(states).cuda()

    # Reconstruct the first timestep
    reconstructed = decoder(encoder(states[:, 0]))
    imutil.show(reconstructed)

if __name__ == '__main__':
    main()
