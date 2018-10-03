import numpy as np
import torch
from torch import nn
from logutil import TimeSeries

class LinearClassifier(nn.Module):
    def __init__(self, input_dim, output_classes):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, output_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.softmax(x, dim=1)
        return x


# simulator: a function that takes a (batch_size, true_latent_dim) np array as input
#   and outputs a (batch_size, channels, height, width) np array of images
# true_latent_dim: integer, true number of generative factors used by simulator
# encoder: a model that takes a (batch_size, channels, height, width) FloatTensor
#   of images and outputs a (batch_size, encoded_latent_dim) tensor
def higgins_metric(simulator, true_latent_dim, encoder, encoded_latent_dim,
                   batch_size=32, train_iters=1000):
    # Train a linear classifier using uniform randomly-generated pairs of images,
    # where the pair shares one generative factor in common.
    # Given the learned encodings of a pair, predict which factor is the same.
    linear_model = LinearClassifier(encoded_latent_dim, true_latent_dim)
    optimizer = torch.optim.Adam(linear_model.parameters())
    ts = TimeSeries('Higgins Metric', train_iters)

    for train_iter in range(train_iters):
        # Generate batch_size pairs
        random_factors = np.random.uniform(size=(batch_size, 2, true_latent_dim))

        # For each pair, select a factor to set
        y_labels = np.random.randint(0, true_latent_dim, size=batch_size)
        for i in range(batch_size):
            y = y_labels[i]
            random_factors[i][0][y] = random_factors[i][1][y]

        # For each pair, generate images with the simulator and encode the images
        images_left = simulator(random_factors[:,0,:])
        images_right = simulator(random_factors[:,1,:])

        # Now encode each pair and take their difference
        encoded_left = encoder(torch.FloatTensor(images_left).unsqueeze(1))
        encoded_right = encoder(torch.FloatTensor(images_right).unsqueeze(1))
        z_diff = encoded_left - encoded_right

        # Now given z_diff, predict y_labels
        optimizer.zero_grad()
        target = torch.LongTensor(y_labels)
        output = linear_model(z_diff)
        y_pred = output.max(1, keepdim=True)[1]
        num_correct = y_pred.eq(target.view_as(y_pred)).sum().item()
        loss = -nn.functional.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        # Track the training accuracy over time
        ts.collect('NLL Loss', loss)
        ts.collect('Train accuracy', num_correct / batch_size)
        ts.print_every(2)
        # Print accuracy for an extra big test batch at the end
        if train_iter == train_iters - 2:
            batch_size = 1000
    print(ts)
    print('Test Accuracy: {}'.format(num_correct / batch_size))

    return num_correct / batch_size

