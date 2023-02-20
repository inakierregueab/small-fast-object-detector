import torch.nn as nn
import torch

from torchvision.transforms import Resize
from torchvision.transforms import InterpolationMode
from models.common import CBL, C3


class PANet(nn.Module):
    def __init__(self, first_out, backbone_connection):
        super(PANet, self).__init__()

        self.backbone_connection = backbone_connection

        self.neck = nn.ModuleList()
        self.neck += [
            CBL(in_channels=first_out*16, out_channels=first_out*8, kernel_size=1, stride=1, padding=0),
            C3(in_channels=first_out*16, out_channels=first_out*8, width_multiple=0.25, depth=2, backbone=False),
            CBL(in_channels=first_out*8, out_channels=first_out*4, kernel_size=1, stride=1, padding=0),
            C3(in_channels=first_out*8, out_channels=first_out*4, width_multiple=0.25, depth=2, backbone=False),
            CBL(in_channels=first_out*4, out_channels=first_out*4, kernel_size=3, stride=2, padding=1),
            C3(in_channels=first_out*8, out_channels=first_out*8, width_multiple=0.5, depth=2, backbone=False),
            CBL(in_channels=first_out*8, out_channels=first_out*8, kernel_size=3, stride=2, padding=1),
            C3(in_channels=first_out*16, out_channels=first_out*16, width_multiple=0.5, depth=2, backbone=False)
        ]

    # TODO: Check skipped connections
    def forward(self, x):
        neck_connection = []
        outputs = []
        for idx, layer in enumerate(self.neck):
            # Connect with backbone
            if idx in [0, 2]:
                x = layer(x)
                neck_connection.append(x)
                x = Resize([x.shape[2] * 2, x.shape[3] * 2], interpolation=InterpolationMode.NEAREST)(x)
                x = torch.cat([x, self.backbone_connection.pop(-1)], dim=1)
            # Neck reconnection
            elif idx in [4, 6]:
                x = layer(x)
                x = torch.cat([x, neck_connection.pop(-1)], dim=1)
            # Output to pass to heads
            elif isinstance(layer, C3) and idx > 2:
                x = layer(x)
                outputs.append(x)
            else:
                x = layer(x)
        return outputs
