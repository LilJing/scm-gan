import torch
from torch import nn


class CoordConv2d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(*args, **kwargs)

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        coord_x = torch.arange(-1.0, 1.0, 2/width).unsqueeze(0).repeat(width,1).unsqueeze(0).repeat(batch_size,1,1).unsqueeze(1).cuda()
        coord_y = torch.arange(-1.0, 1.0, 2/height).unsqueeze(1).repeat(1,height).unsqueeze(0).repeat(batch_size,1,1).unsqueeze(1).cuda()
        x = torch.cat([x, coord_x, coord_y], dim=1)
        return self.conv(x)
