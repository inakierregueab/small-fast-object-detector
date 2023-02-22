import torch.nn as nn
import torch


class Heads(nn.Module):
    def __init__(self, nc=80, anchors=(), ch=(), stride=[8, 16, 32]):
        """
        Parameters:
            nc (int): number of classes
            anchors (list): list of anchors (3x3x2 matrix for 3 detection layers)
            ch (list): list of channels of the input tensors
        """
        super(Heads, self).__init__()
        self.nc = nc    # number of classes
        self.nl = len(anchors)  # number of detection layers
        self.naxs = len(anchors[0])  # number of anchors per scale
        self.stride = stride  # strides computed during build

        # anchors are divided by the stride (anchors_for_head_1/8, anchors_for_head_1/16 etc.)
        anchors_ = torch.tensor(anchors).float().view(self.nl, -1, 2) / torch.tensor(self.stride).repeat(6, 1).T.reshape(3, 3, 2)
        # Store the parameters of the model which should be saved and restored in the state_dict, but are not trained by the optimizer
        self.register_buffer('anchors', anchors_)

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
