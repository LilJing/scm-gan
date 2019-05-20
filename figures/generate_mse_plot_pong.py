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


def plot_mse(plt, mean_filename, err_filename, facecolor='#BBBBFF', edgecolor='#0000FF', label='', already_plotted=False):
    meanvals = np.array(json.load(open(mean_filename)))
    errvals = np.array(json.load(open(err_filename)))

    # Add shaded region to indicate stddev
    x = np.array(range(len(meanvals)))
    if not already_plotted:
        plt.plot(x, meanvals, color=edgecolor, label=label)
    plt.fill_between(x, meanvals - errvals, meanvals + errvals,
                     alpha=0.10, facecolor=facecolor, edgecolor=edgecolor)


input_specs = [
    {
        'title': 'Baseline',
        'mean': '/mnt/nfs/experiments/default/scm-gan_810b894e/mse_default_iter_010000.json',
        'stddev': '/mnt/nfs/experiments/default/scm-gan_810b894e/mse_stddev_default_iter_010000.json',
        'color': '#FF0000',
    },
#    {
#        'title': 'Baseline 2',
#        'mean': '/mnt/nfs/experiments/default/scm-gan_5fe3edee/mse_default_iter_010000.json',
#        'stddev': '/mnt/nfs/experiments/default/scm-gan_5fe3edee/mse_stddev_default_iter_010000.json',
#        'color': '#2222FF',
#    },
    {
        'title': 'Latent Overshooting',
        'mean': '/mnt/nfs/experiments/default/scm-gan_eaa68b1c/mse_default_iter_010000.json',
        'stddev': '/mnt/nfs/experiments/default/scm-gan_eaa68b1c/mse_stddev_default_iter_010000.json',
        'color': '#00FF00',
    },
#    {
#        'title': 'TD adjusted symmetric',
#        'mean': '/mnt/nfs/experiments/default/scm-gan_4b9e8563/mse_default_iter_010000.json',
#        'stddev': '/mnt/nfs/experiments/default/scm-gan_4b9e8563/mse_stddev_default_iter_010000.json',
#        'color': '#0000FF',
#    },
#    {
#        'title': 'TD adjusted symmetric more steps',
#        'mean': '/mnt/nfs/experiments/default/scm-gan_752fd3af/mse_default_iter_010000.json',
#        'stddev': '/mnt/nfs/experiments/default/scm-gan_752fd3af/mse_stddev_default_iter_010000.json',
#        'color': '#551155',
#    },
#    {
#        'title': 'TD adjusted symmetric one-step',
#        'mean': '/mnt/nfs/experiments/default/scm-gan_02273f9b/mse_default_iter_010000.json',
#        'stddev': '/mnt/nfs/experiments/default/scm-gan_02273f9b/mse_stddev_default_iter_010000.json',
#        'color': '#FFFF00',
#    },
#    {
#        'title': 'TD',
#        'mean': '/mnt/nfs/experiments/default/scm-gan_31c2869e/mse_default_iter_010000.json',
#        'stddev': '/mnt/nfs/experiments/default/scm-gan_31c2869e/mse_stddev_default_iter_010000.json',
#        'color': '#2222FF',
#    },
    {
        'title': 'TD',
        'mean': '/mnt/nfs/experiments/default/scm-gan_63175501/mse_default_iter_010000.json',
        'stddev': '/mnt/nfs/experiments/default/scm-gan_63175501/mse_stddev_default_iter_010000.json',
        'color': '#2222FF',
    },

]

def load_data(spec):
    result = {}
    result['title'] = spec['title']
    result['mean'] = np.array(json.load(open(spec['mean'])))
    result['stddev'] = np.array(json.load(open(spec['stddev'])))
    return result

data_dicts = [load_data(spec) for spec in input_specs]
colors = ['#FF0000', '#00FF00', '#0000FF']


# Hack: Plot the first series just to get an axis, THEN plot all the rest
plot_params = {
    'title': 'Mean Squared Error: Pong',
    'grid': True,
    'label': input_specs[0]['title'],
    'color': input_specs[0]['color'],
}
plt = pd.Series(data_dicts[0]['mean']).plot(**plot_params)
plt.set_ylim(ymin=0)
plt.set_ylabel('Pixel MSE')
plt.set_xlabel('Prediction horizon (timesteps)')

# Now plot all the lines
for spec in input_specs:
    color = spec['color']
    already_plotted = spec['title'] == input_specs[0]['title']
    plot_mse(plt, spec['mean'], spec['stddev'], facecolor=color, edgecolor=color, label=spec['title'], already_plotted=already_plotted)

plt.legend(loc='bottom right')

filename = 'mse_graph.png'
imutil.show(plt, filename=filename)
pyplot.close()
