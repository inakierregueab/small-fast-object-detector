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
        # Head returns a list of 3 tensors (one per head) with shape (batch_size, predictions_per_scale, grid_y, grid_x, 5 + nc)
        x = self.heads(x)
        return x

    def cells_to_bboxes(self, predictions, anchors, strides):
        num_out_layers = len(predictions)
        grid = [torch.empty(0) for _ in range(num_out_layers)]  # initialize
        anchor_grid = [torch.empty(0) for _ in range(num_out_layers)]  # initialize
        all_bboxes = []
        for i in range(num_out_layers):
            bs, naxs, ny, nx, _ = predictions[i].shape
            stride = strides[i]
            grid[i], anchor_grid[i] = self.make_grids(anchors, naxs, ny=ny, nx=nx, stride=stride, i=i)

            # TODO: sigmoid applied element-wise, why not use softmax for class prediction?
            layer_prediction = predictions[i].sigmoid()

            obj = layer_prediction[..., 4:5]
            xy = (2 * (layer_prediction[..., 0:2]) + grid[i] - 0.5) * stride
            wh = ((2 * layer_prediction[..., 2:4]) ** 2) * anchor_grid[i]
            best_class = torch.argmax(layer_prediction[..., 5:], dim=-1).unsqueeze(-1)

            scale_bboxes = torch.cat((best_class, obj, xy, wh), dim=-1).reshape(bs, -1, 6)
            all_bboxes.append(scale_bboxes)

        return torch.cat(all_bboxes, dim=1)

    def make_grids(self, anchors, naxs, stride, nx=20, ny=20, i=0):
        x_grid = torch.arange(nx)
        x_grid = x_grid.repeat(ny).reshape(ny, nx)
        y_grid = torch.arange(ny).unsqueeze(0)
        y_grid = y_grid.T.repeat(1, nx).reshape(ny, nx)
        xy_grid = torch.stack([x_grid, y_grid], dim=-1)
        xy_grid = xy_grid.expand(1, naxs, ny, nx, 2)

        anchor_grid = (anchors[i] * stride).reshape((1, naxs, 1, 1, 2)).expand(1, naxs, ny, nx, 2)

        return xy_grid, anchor_grid


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





