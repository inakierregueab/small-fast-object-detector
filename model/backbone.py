import torch.nn as nn
import torch

from model.common import CBL, C3


class SPPF(nn.Module):
    """
    Spatial Pyramid Pooling Feature - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    Parameters:
        in_channels (int): number of channel of the input tensor
        out_channels (int): number of channel of the output tensor
    """
    def __init__(self, in_channels, out_channels):
        super(SPPF, self).__init__()

        c_ = int(in_channels//2)

        self.c1 = CBL(in_channels, c_, 1, 1, 0)
        self.pool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.c_out = CBL(c_ * 4, out_channels, 1, 1, 0)

    def forward(self, x):
        x = self.c1(x)
        pool1 = self.pool(x)
        pool2 = self.pool(pool1)
        pool3 = self.pool(pool2)

        # TODO: check concatenation
        return self.c_out(torch.cat([x, pool1, pool2, pool3], dim=1))


class CSPBackbone(nn.Module):
    """
    Parameters:
        first_out (int): number of channel of the first output tensor
    """
    def __init__(self, first_out):
        super(CSPBackbone, self).__init__()

        self.backbone = nn.ModuleList()
        self.backbone += [
            CBL(in_channels=3, out_channels=first_out, kernel_size=6, stride=2, padding=2),
            CBL(in_channels=first_out, out_channels=first_out * 2, kernel_size=3, stride=2, padding=1),
            C3(in_channels=first_out * 2, out_channels=first_out * 2, width_multiple=0.5, depth=2),
            CBL(in_channels=first_out * 2, out_channels=first_out * 4, kernel_size=3, stride=2, padding=1),
            # next out
            C3(in_channels=first_out * 4, out_channels=first_out * 4, width_multiple=0.5, depth=4),
            CBL(in_channels=first_out * 4, out_channels=first_out * 8, kernel_size=3, stride=2, padding=1),
            # next out
            C3(in_channels=first_out * 8, out_channels=first_out * 8, width_multiple=0.5, depth=6),
            CBL(in_channels=first_out * 8, out_channels=first_out * 16, kernel_size=3, stride=2, padding=1),
            C3(in_channels=first_out * 16, out_channels=first_out * 16, width_multiple=0.5, depth=2),
            SPPF(in_channels=first_out * 16, out_channels=first_out * 16)
        ]

    # TODO: modify backbone to return intermediate feature maps (maybe in functional style)
    def forward(self, x):
        backbone_connection = []
        for idx, layer in enumerate(self.backbone):
            # pass through all backbone layers
            x = layer(x)
            # takes the out of the 2nd and 3rd C3 block and stores it for neck passing
            if idx in [4, 6]:
                backbone_connection.append(x)
        return x, backbone_connection



