import random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import imutil


# We don't want eg. Xavier initialization here
# The RNN gradient should explode, not vanish
def init_conv_weight(conv):
    #conv.weight.data.normal_(0, 1)
    pass

def init_gru_weight(gru):
    #gru.weight_hh_l0.data.normal_(0, 1)
    #gru.weight_ih_l0.data.normal_(0, 1)
    pass


class CSRN(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.rnn_in = channels
        self.rnn_out = self.rnn_in
        self.conv_down = nn.Conv1d(self.rnn_out, self.rnn_in, kernel_size=3, stride=1, padding=1)
        self.conv_up = nn.Conv1d(self.rnn_out, self.rnn_in, kernel_size=3, stride=1, padding=1)
        self.conv_left = nn.Conv1d(self.rnn_out, self.rnn_in, kernel_size=3, stride=1, padding=1)
        self.conv_right = nn.Conv1d(self.rnn_out, self.rnn_in, kernel_size=3, stride=1, padding=1)
        self.rnn_down = nn.GRU(self.rnn_in, self.rnn_out, bias=False)
        self.rnn_up = nn.GRU(self.rnn_in, self.rnn_out, bias=False)
        self.rnn_left = nn.GRU(self.rnn_in, self.rnn_out, bias=False)
        self.rnn_right = nn.GRU(self.rnn_in, self.rnn_out, bias=False)
        self.conv_combine = nn.Conv2d(self.rnn_in * 4, channels, kernel_size=1)

        init_conv_weight(self.conv_down)
        init_conv_weight(self.conv_up)
        init_conv_weight(self.conv_left)
        init_conv_weight(self.conv_right)
        init_gru_weight(self.rnn_down)
        init_gru_weight(self.rnn_up)
        init_gru_weight(self.rnn_left)
        init_gru_weight(self.rnn_right)

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        assert channels == self.channels
        # x.shape: (batch, channels, height, width)
        context_above = torch.zeros((batch_size, self.rnn_out, height, width)).cuda()
        context_below = torch.zeros((batch_size, self.rnn_out, height, width)).cuda()
        context_left = torch.zeros((batch_size, self.rnn_out, height, width)).cuda()
        context_right = torch.zeros((batch_size, self.rnn_out, height, width)).cuda()

        # For each row, top to bottom:
        #  Take the output of the RNN on the previous row
        #  Apply a 1D convolution to that row (so activations spread like a cone)
        #  Feed the convolved activations back into the RNN
        rnn_state = torch.zeros((1, batch_size * width, self.rnn_out)).cuda()
        rnn_cx = torch.zeros((1, batch_size * width, self.rnn_out)).cuda()
        for i in range(height):
            pixel_row = x[:, :, i, :]
            pixel_row = pixel_row.permute(0, 2, 1).contiguous()
            pixel_row = pixel_row.view(1, batch_size * width, self.rnn_in)
            rnn_out, rnn_state = self.rnn_down(pixel_row, rnn_state)
            conv_in = rnn_out.view(batch_size, width, self.rnn_out)
            conv_in = conv_in.permute(0, 2, 1)
            context_above[:, :, i, :] = conv_in
            conv_out = self.conv_down(conv_in)
            conv_out = torch.tanh(conv_out)
            rnn_state = conv_out.permute(0, 2, 1).contiguous()
            rnn_state = rnn_state.view(1, batch_size * width, self.rnn_in)

        rnn_state = torch.zeros((1, batch_size * width, self.rnn_out)).cuda()
        for i in reversed(range(height)):
            pixel_row = x[:, :, i, :]
            pixel_row = pixel_row.permute(0, 2, 1).contiguous()
            pixel_row = pixel_row.view(1, batch_size * width, self.rnn_in)
            rnn_out, rnn_state = self.rnn_up(pixel_row, rnn_state)
            conv_in = rnn_out.view(batch_size, width, self.rnn_out)
            conv_in = conv_in.permute(0, 2, 1)
            context_below[:, :, i, :] = conv_in
            conv_out = self.conv_up(conv_in)
            conv_out = torch.tanh(conv_out)
            rnn_state = conv_out.permute(0, 2, 1).contiguous()
            rnn_state = rnn_state.view(1, batch_size * width, self.rnn_out)

        rnn_state = torch.zeros((1, batch_size * height, self.rnn_out)).cuda()
        for i in range(width):
            pixel_col = x[:, :, :, i]
            pixel_col = pixel_col.permute(0, 2, 1).contiguous()
            pixel_col = pixel_col.view(1, batch_size * height, self.rnn_in)
            rnn_out, rnn_state = self.rnn_left(pixel_col, rnn_state)
            conv_in = rnn_out.view(batch_size, height, self.rnn_out)
            conv_in = conv_in.permute(0, 2, 1)
            context_left[:, :, :, i] = conv_in
            conv_out = self.conv_left(conv_in)
            conv_out = torch.tanh(conv_out)
            rnn_state = conv_out.permute(0, 2, 1).contiguous()
            rnn_state = rnn_state.view(1, batch_size * height, self.rnn_out)

        rnn_state = torch.zeros((1, batch_size * height, self.rnn_out)).cuda()
        for i in reversed(range(width)):
            pixel_col = x[:, :, :, i]
            pixel_col = pixel_col.permute(0, 2, 1).contiguous()
            pixel_col = pixel_col.view(1, batch_size * height, self.rnn_in)
            rnn_out, rnn_state = self.rnn_right(pixel_col, rnn_state)
            conv_in = rnn_out.view(batch_size, height, self.rnn_out)
            conv_in = conv_in.permute(0, 2, 1)
            context_left[:, :, :, i] = conv_in
            conv_out = self.conv_right(conv_in)
            conv_out = torch.tanh(conv_out)
            rnn_state = conv_out.permute(0, 2, 1).contiguous()
            rnn_state = rnn_state.view(1, batch_size * height, self.rnn_out)

        context_map = torch.cat((context_above, context_below, context_left, context_right), dim=1)
        output = self.conv_combine(context_map)
        #output = torch.sigmoid(output)
        return output


class SimpleFCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(3,3), padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.csrn1 = CSRN(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 3, kernel_size=(3,3), padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.leaky_relu(x, 0.2)
        x = self.bn1(x)
        x = self.csrn1(x)
        x = self.bn2(x)
        x = F.leaky_relu(x, 0.2)
        x = self.conv2(x)
        x = torch.sigmoid(x)
        return x


if __name__ == '__main__':
    def to_tensor(img):
        x = np.array([img])
        x = np.moveaxis(x, -1, 1)
        x = torch.Tensor(x)
        return x.cuda()

    kitty = imutil.show('small_kitty.jpg', return_pixels=True, save=False)
    kitty /= 255.
    def get_example_pair():
        canvas = np.ones((128, 128, 3))
        target_canvas = np.ones((128, 128, 3))
        #rx = random.randint(0,64)
        #ry = random.randint(20,64)
        rx = 40
        ry = 20
        canvas[ry:ry+64, rx:rx+54] = kitty
        target_canvas[ry+64:, rx:rx+54] = (1,0,0)
        target_canvas[ry:ry+64, :rx] = (0,1,0)
        target_canvas[ry:ry+64, rx+54:] = (0,0,1)

        return canvas, target_canvas


    def get_batch():
        x, y = get_example_pair()
        x, y = to_tensor(x), to_tensor(y)
        return x, y

    model = SimpleFCN()
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters())
    for i in range(1000):
        optimizer.zero_grad()
        x, y = get_batch()
        output = model(x)
        mse_loss = torch.mean((output - y) ** 2)
        mse_loss.backward()
        print('Loss: {}'.format(mse_loss))
        optimizer.step()
        #imutil.show(x[0], save=False)
        #imutil.show(y[0], save=False)
        imutil.show(output[0], video_filename='kitty.mjpeg', caption='K3 w/bias step {}'.format(i), font_size=12)

