import numpy as np
import torch
from torch import nn
from torch import functional as F
import imutil


class CSRN(nn.Module):
    def __init__(self, height, width, channels):
        super().__init__()
        self.channels = channels
        self.gru_in = 3
        self.gru_out = 3
        self.conv_rows = nn.Conv1d(channels, self.gru_in, kernel_size=3, stride=1, padding=1)
        self.conv_cols = nn.Conv1d(channels, self.gru_in, kernel_size=3, stride=1, padding=1)
        self.gru_rows = nn.GRU(self.gru_in, self.gru_out)
        self.gru_cols = nn.GRU(self.gru_in, self.gru_out)
        self.conv_combine = nn.Conv2d(self.gru_out * 2, channels, kernel_size=1)
        #self.conv_rows.weight.data += 1

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        assert channels == self.channels
        # x.shape: (batch, channels, height, width)
        row_wise = torch.zeros((batch_size, height, width, self.gru_out))
        width_h0 = torch.zeros((1, batch_size * width, self.gru_out))
        col_wise = torch.zeros((batch_size, height, width, self.gru_out))
        height_h0 = torch.zeros((1, batch_size * height, self.gru_out))

        # Run vertically
        for i in range(height):
            rows_in = x[:, :, i]
            # (batch, channels, width)
            conv_row = self.conv_rows(rows_in)
            conv_row = torch.tanh(conv_row)
            # (batch, self.gru_in, width)
            gru_input = conv_row.permute(0,2,1).contiguous().view(-1, self.gru_in)
            # (batch * width, self.gru_in)
            gru_input = gru_input.unsqueeze(0)
            # (1, batch * width, self.gru_in)
            rnn_out, width_h0 = self.gru_rows(gru_input, width_h0)
            row_wise[:, i] = rnn_out.view(batch_size, width, self.gru_out)
        imutil.show(row_wise[0])

        # Run horizontally
        for i in range(width):
            cols_in = x[:, :, :, i]
            # (batch, channels, height)
            conv_col = self.conv_cols(cols_in)
            conv_col = torch.tanh(conv_col)

            # (batch, self.gru_in, height)
            gru_input = conv_col.permute(0,2,1).contiguous().view(-1, self.gru_in)
            # (batch * height, self.gru_in)
            gru_input = gru_input.unsqueeze(0)
            # (1, batch * height, self.gru_in)
            rnn_out, height_h0 = self.gru_cols(gru_input, height_h0)
            col_wise[:, :, i] = rnn_out.view(batch_size, height, self.gru_out)
        imutil.show(col_wise[0])

        gru_output = torch.cat((row_wise, col_wise), dim=-1)
        gru_output = gru_output.permute((0, 3, 1, 2))
        return torch.sigmoid(self.conv_combine(gru_output))


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
