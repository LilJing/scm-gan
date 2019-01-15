# Headless matplotlib fix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import numpy as np
import networkx as nx
import imutil


# This function identifies the non-pruned connections
def compute_causal_graph(model, latent_size, num_actions):
    # TODO: fix for outer product
    return np.eye(latent_size)

    input_size = latent_size + num_actions
    rows = []
    for i in range(input_size):
        x = torch.zeros(input_size)
        W1 = model.fc1.weight.abs().cpu()
        W2 = model.fc2.weight.abs().cpu()

        # Continuous version
        zero_return = torch.matmul(W2, torch.matmul(W1, x))# + model.fc2.bias
        x[i] = 1.
        one_return = torch.matmul(W2, torch.matmul(W1, x))# + model.fc2.bias
        result = (one_return - zero_return)

        """
        # Thresholded version
        def binarize(W, theta=.01):
            return (W.abs() > theta).type(torch.FloatTensor)
        result = torch.matmul(binarize(model.fc2.weight), torch.matmul(binarize(model.fc1.weight), x))
        """
        rows.append(np.array(result.abs().cpu().data))

    scm = np.array(rows)
    scm -= scm.min()
    eps = .0001
    return scm / (scm.max() + eps)


# Just a helper function to render the graph to an image
def render_causal_graph(scm):

    plt.cla()

    # The SCM will have more rows than columns
    # Pad with zeros to create a square adjacency matrix
    rows, cols = scm.shape
    adjacency = np.zeros((rows, rows))
    adjacency[:,:cols] = scm[:]

    edge_alphas = adjacency.flatten()

    from networkx.classes.multidigraph import DiGraph
    G = DiGraph(np.ones(adjacency.shape))

    pos = nx.layout.circular_layout(G)

    node_sizes = [10 for i in range(len(G))]
    M = G.number_of_edges()
    #edge_colors = range(2, M + 2)
    #edge_alphas = [(5 + i) / (M + 4) for i in range(M)]
    #edge_colors = [2 for i in range(len(G))]

    nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='blue')
    edges = nx.draw_networkx_edges(G, pos, node_size=node_sizes, arrowstyle='->', arrowsize=20, edge_cmap=plt.cm.Blues, width=2)
    labels = ['$z_{{{}}}$'.format(i) for i in range(cols)] + ['$a_{{{}}}$'.format(i) for i in range(rows - cols)]
    labels = {i: labels[i] for i in range(len(labels))}
    pos = {k: (v[0], v[1] + .1) for (k,v) in pos.items()}
    nx.draw_networkx_labels(G, pos, labels, font_size=16)
    # set alpha value for each edge
    for i in range(M):
        edges[i].set_alpha(edge_alphas[i])
        ax = plt.gca()
        ax.set_axis_off()
        plt.show()
    return imutil.show(plt, return_pixels=True, display=False, save=False)
