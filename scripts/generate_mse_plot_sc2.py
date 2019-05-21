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
eval_at_iter = 150000
plot_title = 'StarIntruders, Stochastic Model'

input_specs = [
    # Stochastic models (bernoulli)
    {
        'title': 'BPTT +L1',
        'experiment_id': 'scm-gan_547306a3',
        'color': '#2222FF',
    },
    {
        'title': 'BPTT +L1 +TD',
        'experiment_id': 'scm-gan_531331ef',
        'color': '#000055',
    },
    {
        'title': 'BPTT +L1 +LO',
        'experiment_id': 'scm-gan_a8a9d765',
        'color': '#FF2222',
    },
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
