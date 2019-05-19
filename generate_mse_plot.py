import argparse

import time
import math
import os
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import imutil
import pandas as pd
from matplotlib import pyplot


def plot_mse(plt, mean_filename, err_filename, facecolor='#BBBBFF', edgecolor='#0000FF'):
    meanvals = np.array(json.load(open(mean_filename)))
    errvals = np.array(json.load(open(err_filename)))

    # Add shaded region to indicate stddev
    x = np.array(range(len(meanvals)))
    plt.plot(x, meanvals, color=edgecolor)
    plt.fill_between(x, meanvals - errvals, meanvals + errvals,
                     alpha=0.2, facecolor=facecolor, edgecolor=edgecolor)


# Usage: generate_mse_plot.py /path/to/experiment1/mse_iter_1000.json /path/to/experiment2/mse_iter_1000.json
data_files = sys.argv[1:]

data_files = [
    '/mnt/nfs/experiments/default/scm-gan_649a849f/mse_iter_099000.json',
]


mse_losses = np.array(json.load(open(data_files[0])))
plot_params = {
    'title': 'Mean Squared Error Pixel Loss',
    'grid': True,
}
plt = pd.Series(mse_losses).plot(**plot_params)
plt.set_ylim(ymin=0)
plt.set_ylabel('Pixel MSE')
plt.set_xlabel('Prediction horizon (timesteps)')

plot_mse(plt, mse_filename, stddev_filename, facecolor='0000FF', edgecolor='0000FF')
plot_mse(plt, mse_filename, stddev_filename, facecolor='00FF00', edgecolor='00FF00')
plot_mse(plt, mse_filename, stddev_filename, facecolor='FF0000', edgecolor='FF0000')

filename = 'mse_graph.png'
imutil.show(plt, filename=filename)
pyplot.close()
