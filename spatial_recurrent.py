import numpy as np
import torch
from torch import nn
from torch import functional as F
import imutil


class CSRN(nn.Module):
    def __init__(self, height, width, channels):
        super().__init__()
        self.channels = channels
        self.rnn_in = channels
        self.rnn_out = 3
        self.conv_rows = nn.Conv1d(self.rnn_out, self.rnn_in, kernel_size=3, stride=1, padding=1)
        self.conv_cols = nn.Conv1d(self.rnn_out, self.rnn_in, kernel_size=3, stride=1, padding=1)
        self.rnn_rows = nn.GRU(self.rnn_in, self.rnn_out)
        self.rnn_cols = nn.GRU(self.rnn_in, self.rnn_out)
        self.conv_combine = nn.Conv2d(self.rnn_out * 2, channels, kernel_size=1)
        #self.conv_rows.weight.data += 1
        #self.conv_rows.weight.data[-1][:2] = 0
        #self.conv_cols.weight.data[-1][:2] = 0

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        assert channels == self.channels
        # x.shape: (batch, channels, height, width)
        context_above = torch.zeros((batch_size, self.rnn_out, height, width))
        context_left = torch.zeros((batch_size, self.rnn_out, height, width))

        # For each row, top to bottom:
        #  Take the output of the RNN on the previous row
        #  Apply a 1D convolution to that row (so activations spread like a cone)
        #  Feed the convolved activations back into the RNN
        rnn_state = torch.zeros((1, batch_size * width, self.rnn_in))
        for i in range(height):
            pixel_row = x[:, :, i, :]
            pixel_row = pixel_row.permute(0, 2, 1).contiguous()
            pixel_row = pixel_row.view(1, batch_size * width, self.rnn_in)
            rnn_out, rnn_state = self.rnn_rows(pixel_row, rnn_state)
            conv_in = rnn_out.view(batch_size, width, self.rnn_in)
            conv_in = conv_in.permute(0, 2, 1)
            context_above[:, :, i, :] = conv_in
            conv_out = self.conv_rows(conv_in)
            conv_out = torch.tanh(conv_out)
            rnn_state = conv_out.permute(0, 2, 1).contiguous()
            rnn_state = rnn_state.view(1, batch_size * width, self.rnn_in)

        rnn_state = torch.zeros((1, batch_size * height, self.rnn_in))
        for i in range(width):
            pixel_col = x[:, :, :, i]
            pixel_col = pixel_col.permute(0, 2, 1).contiguous()
            pixel_col = pixel_col.view(1, batch_size * height, self.rnn_in)
            rnn_out, rnn_state = self.rnn_cols(pixel_col, rnn_state)
            conv_in = rnn_out.view(batch_size, height, self.rnn_in)
            conv_in = conv_in.permute(0, 2, 1)
            context_left[:, :, :, i] = conv_in
            conv_out = self.conv_cols(conv_in)
            conv_out = torch.tanh(conv_out)
            rnn_state = conv_out.permute(0, 2, 1).contiguous()
            rnn_state = rnn_state.view(1, batch_size * height, self.rnn_in)
        import pdb; pdb.set_trace()

        context_map = torch.cat((context_above, context_left), dim=1)
        return torch.sigmoid(self.conv_combine(context_map))


img = imutil.show('kittenpuppy.jpg', return_pixels=True)
img -= 128.
img /= 255.
x = np.array([img] * 16)
x = np.moveaxis(x, -1, 1)
batch_size, channels, height, width = x.shape
x = torch.Tensor(x)
model = CSRN(height, width, channels)
output = model(x)

imutil.show(output[0])
