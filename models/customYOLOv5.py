# Inspired by: https://github.com/AlessandroMondin/YOLOV5m

import torch.nn as nn
import torch

from backbone import CSPBackbone
from neck import PANet


class YOLOv5m(nn.Module):
    def __init__(self, first_out, nc=80, anchors=(), ch=(), inference=False):
        super(YOLOv5m, self).__init__()
        # TODO: inference?
        self.inference = inference
        self.backbone = CSPBackbone(first_out=first_out)
        self.neck = PANet(first_out=first_out)
        # To be done
        self.head = lambda x: x

    def forward(self, x):
        x, backbone_connection = self.backbone(x)
        x = self.neck(x, backbone_connection)
        x = self.head(x)
        return x



