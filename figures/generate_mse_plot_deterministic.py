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



file_prefix = '/mnt/nfs/experiments/default/'
eval_at_iter = 10000
plot_title = 'Pong, Deterministic Model'

input_specs = [
    # Stochastic models (bernoulli)
#    {
#        'title': 'BPTT +L1',
#        'experiment_id': 'scm-gan_810b894e',
#        'color': '#FF0000',
#    },
#    {
#        'title': 'BPTT +L1 +LO',
#        'experiment_id': 'scm-gan_eaa68b1c',
#        'color': '#00FF00',
#    },
#    {
#        'title': 'BPTT +L1 +TD',
#        'experiment_id': 'scm-gan_63175501',
#        'color': '#2222FF',
#    },
    # Deterministic models
    {
        'title': 'BPTT (baseline)',
        'experiment_id': 'scm-gan_3c02865d',
        'color': '#550000',
    },
    {
        'title': 'BPTT +LO',
        'experiment_id': 'scm-gan_65022c52',
        'color': '#005500',
    },
    {
        'title': 'BPTT +TD',
        'experiment_id': 'scm-gan_6788ea40',
        'color': '#000055',
    },
    {
        'title': 'BPTT +L1',
        'experiment_id': 'scm-gan_9e128ad8',
        'color': '#FF2222',
    },
#    {
#        'title': 'BPTT +L1 + LO',
#        'experiment_id': 'scm-gan_d788b8f3',
#        'color': '#22FF22',
#    },
#    {
#        'title': 'BPTT +L1 +TD',
#        'experiment_id': 'scm-gan_bd9540a5',
#        'color': '#2222FF',
#    },
]

def load_data(spec):
    result = spec.copy()
    experiment_id = spec['experiment_id']
    mse_filename = '{}/{}/mse_default_iter_{:06d}.json'.format(file_prefix, experiment_id, eval_at_iter)
    std_filename = '{}/{}/mse_stddev_default_iter_{:06d}.json'.format(file_prefix, experiment_id, eval_at_iter)
    result['mean'] = np.array(json.load(open(mse_filename)))
    result['stddev'] = np.array(json.load(open(std_filename)))
    return result


def plot_mse(plt, meanvals, errvals, facecolor='#BBBBFF', edgecolor='#0000FF', label='', already_plotted=False):
    # Add shaded region to indicate stddev
    x = np.array(range(len(meanvals)))
    if not already_plotted:
        plt.plot(x, meanvals, color=edgecolor, label=label)
    plt.fill_between(x, meanvals - errvals, meanvals + errvals,
                     alpha=0.10, facecolor=facecolor, edgecolor=edgecolor)


data_dicts = [load_data(spec) for spec in input_specs]

# Hack: Plot the first series just to get an axis, THEN plot all the rest
plot_params = {
    'title': plot_title,
    'grid': True,
    'label': input_specs[0]['title'],
    'color': input_specs[0]['color'],
}
plt = pd.Series(data_dicts[0]['mean']).plot(**plot_params)
plt.set_ylim(ymin=0)
plt.set_ylabel('Pixel MSE')
plt.set_xlabel('Prediction horizon (timesteps)')

# Now plot all the lines
for spec in data_dicts:
    color = spec['color']
    already_plotted = spec['title'] == input_specs[0]['title']
    plot_mse(plt, spec['mean'], spec['stddev'], facecolor=color, edgecolor=color, label=spec['title'], already_plotted=already_plotted)

plt.legend(loc='best')

filename = 'mse_graph.png'
imutil.show(plt, filename=filename)
pyplot.close()
