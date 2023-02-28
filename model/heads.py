import torch.nn as nn


class Heads(nn.Module):
    def __init__(self, nc, stride, anchors, ch):
        """
        Parameters:
            nc (int): number of classes
            anchors (list): list of anchors (3x3x2 matrix for 3 detection layers)
            ch (list): list of channels of the input tensors
        """
        super(Heads, self).__init__()
        self.nc = nc # number of classes
        self.anchors = anchors  # anchors per detection layer
        self.nl = len(anchors)  # number of detection layers
        self.naxs = len(anchors[0])  # number of anchors per scale
        self.stride = stride  # strides computed during build

        self.out_convs = nn.ModuleList()
        for in_channels in ch:
            self.out_convs += [
                nn.Conv2d(in_channels=in_channels, out_channels=(5 + self.nc) * self.naxs, kernel_size=1)
            ]

    def forward(self, x):
        """
        Returns a list of 3 tensors (one per head) with shape (batch_size, predictions_per_scale, grid_y, grid_x, 5 + nc)
        """
        for i in range(self.nl):
            x[i] = self.out_convs[i](x[i])
            bs, _, grid_y, grid_x = x[i].shape
            # Permutation as “legacy from YOLOv3"
            x[i] = x[i].view(bs, self.naxs, (5 + self.nc), grid_y, grid_x).permute(0, 1, 3, 4, 2).contiguous()
        return x
