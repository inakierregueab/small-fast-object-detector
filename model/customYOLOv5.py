import torch.nn as nn
import torch
import time

from backbone import CSPBackbone
from neck import PANet
from heads import Heads
from base import BaseModel


# Inspired by: https://github.com/AlessandroMondin/YOLOV5m
class YOLOv5m(BaseModel):
    def __init__(self, first_out, nc=80, anchors=(), ch=(), inference=False):
        super(YOLOv5m, self).__init__()
        # TODO: inference?
        self.inference = inference
        self.backbone = CSPBackbone(first_out=first_out)
        self.neck = PANet(first_out=first_out)
        self.heads = Heads(nc=nc, anchors=anchors, ch=ch)

    def forward(self, x):
        x, backbone_connection = self.backbone(x)
        x = self.neck(x, backbone_connection)
        x = self.heads(x)
        return x


if __name__ == "__main__":

    # Parameters
    batch_size = 2
    image_height = 640
    image_width = 640
    nc = 80
    anchors = [
        [(10, 13), (16, 30), (33, 23)],  # P3/8
        [(30, 61), (62, 45), (59, 119)],  # P4/16
        [(116, 90), (156, 198), (373, 326)]  # P5/32
        ]
    first_out = 48

    # Random input
    x = torch.rand(batch_size, 3, image_height, image_width)

    model = YOLOv5m(first_out=first_out, nc=nc, anchors=anchors,
                    ch=(first_out*4, first_out*8, first_out*16), inference=False)

    start = time.time()
    out = model(x)
    end = time.time()

    assert out[0].shape == (batch_size, 3, image_height//8, image_width//8, nc + 5)
    assert out[1].shape == (batch_size, 3, image_height//16, image_width//16, nc + 5)
    assert out[2].shape == (batch_size, 3, image_height//32, image_width//32, nc + 5)

    print("Success!")
    print("feedforward took {:.2f} seconds".format(end - start))

    # Check model size
    print("Total parameters: {:.5f}M".format(sum(p.numel() for p in model.parameters()) / 1e6))
    print("Total trainable parameters: {:.5f}M".format(sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6))
    # Add buffer to parameters
    print("Size of the model: {:.5f}MB".format(sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2))
    # prints more model info inherited from BaseModel
    # print(model.__str__())





