import torch.nn as nn
import torch
import time

from torchvision.ops import boxes as box_ops

from backbone import CSPBackbone
from neck import PANet
from heads import Heads
from base import BaseModel


# Inspired by: https://github.com/AlessandroMondin/YOLOV5m &
# https://pub.towardsai.net/yolov5-m-implementation-from-scratch-with-pytorch-c8f84a66c98b
class YOLOv5m(BaseModel):
    def __init__(self, first_out, nc=80, anchors=(), ch=(), stride=[8, 16, 32], inference=False):
        super(YOLOv5m, self).__init__()
        self.first_out = first_out
        self.nc = nc  # number of classes

        self.num_heads = len(stride)
        # TODO: why stride anchors?
        # anchors are divided by the stride (anchors_for_head_1/8, anchors_for_head_1/16 etc.)
        anchors_ = torch.tensor(anchors).float().view(self.num_heads, -1, 2) / torch.tensor(stride).repeat(6,1).T.reshape(3,3,2)
        # Store the parameters of the model which should be saved and restored in the state_dict, but are not trained by the optimizer
        self.register_buffer('anchors', anchors_)

        self.ch = ch
        self.stride = stride
        self.inference = inference  # TODO: what is inference?

        self.backbone = CSPBackbone(first_out=self.first_out)
        self.neck = PANet(first_out=self.first_out)
        self.heads = Heads(nc=self.nc, anchors=self.anchors, ch=self.ch, stride=self.stride)

    def forward(self, x):
        """
        Parameters:
            x (tensor): input tensor with shape (batch_size, 3, 640, 640)
        Returns:
            x (list): list of 3 tensors (one per head) with shape (batch_size, predictions_per_scale, grid_y, grid_x, 5 + nc)
        """
        x, backbone_connection = self.backbone(x)
        x = self.neck(x, backbone_connection)
        x = self.heads(x)
        return x

    def postprocessing(self, yolo_output, conf_thres=0.5, iou_thres=0.5):
        """
        Parameters:
            yolo_output (list): list of 3 tensors (one per head) with shape (batch_size, predictions_per_scale, grid_y, grid_x, 5 + nc)
            conf_thres (float): confidence threshold
            iou_thres (float): IoU threshold
        Returns:
            bboxes (list): list of 3 tensors (one per head) with shape (batch_size, detected_objects, 5 + class_idx)
        """
        bboxes = []

        # Iterate over every head/scale output
        for head in range(self.num_heads):
            bs, naxs, ny, nx, _ = yolo_output[head].shape   # get dimensions of the head output

            # Apply sigmoid to the output: in the class section, softmax is not applied because the YOLOv5 model is
            # designed to perform multi-label classification, which means that each object in an image can be
            # associated with multiple labels. Therefore, for each bounding box, the model outputs the probabilities
            # of the object belonging to each class separately, without the constraint that the sum of probabilities
            # across all classes should be equal to 1.
            normalized_output = yolo_output[head].sigmoid()


            # Obtain cxcy grid
            x_grid = torch.arange(nx)
            x_grid = x_grid.repeat(ny).reshape(ny, nx)
            y_grid = torch.arange(ny).unsqueeze(0)
            y_grid = y_grid.T.repeat(1, nx).reshape(ny, nx)
            xy_grid = torch.stack([x_grid, y_grid], dim=-1)
            # Per each anchor in the scale, create an identical grid of (x, y) coordinates
            xy_grid = xy_grid.expand(1, naxs, ny, nx, 2)


            # Obtain anchor grid for every anchor in the scale and undo the stride normalization in __init__
            anchor_grid = (self.anchors[head]*self.stride[head]).reshape((1, naxs, 1, 1, 2)).expand(1, naxs, ny, nx, 2)


            # Obtain the objectness score and the class prediction
            obj = normalized_output[..., 4:5]
            class_pred = normalized_output[..., 5:]
            # TODO: Obtain the class with the highest score multplied by the objectness score or not?
            class_idx = torch.argmax(class_pred, dim=-1).unsqueeze(-1)
            # best_class = torch.argmax(class_pred*obj, dim=-1).unsqueeze(-1)


            # Compute the absolute coordinates of bboxes
            xy = (2 * (normalized_output[..., 0:2]) + xy_grid - 0.5) * self.stride[head]
            wh = ((2 * normalized_output[..., 2:4]) ** 2) * anchor_grid


            # Concat and append to other heads predictions
            head_output = torch.cat((class_idx, obj, xy, wh), dim=-1).reshape(bs, -1, 6)
            bboxes.append(head_output)

        # Concatenate the predictions of all heads, this should be equal to other method
        # Its shape should be (batch_size, N, 6), where N = sum_heads(anchor_per_head * grid_y_head * grid_x_head)
        bboxes = torch.cat(bboxes, dim=1)

        # Apply NMS
        bboxes = self.nms(bboxes, conf_thres=conf_thres, iou_thres=iou_thres)

        return bboxes

    def nms(self, batch_predictions: torch.Tensor, conf_thres: float = 0.0, iou_thres: float = 0.0):
        """
        Applies non-maximum suppression to the predicted boxes.
        """
        # TODO: Avoid for loop, play with pytorch filtering and slicing
        batch_selected_predictions = []
        for prediction in batch_predictions:
            # Extract the objectness/confidence score and class idx
            confidence_score = prediction[..., 1]
            class_idx = prediction[..., 0]

            # Filter out low-confidence boxes
            mask = confidence_score >= conf_thres
            boxes = prediction[mask, 2:]
            scores = confidence_score[mask]
            class_ids = class_idx[mask]

            # Convert the coordinates from (x_center, y_center, width, height) to (x_min, y_min, x_max, y_max)
            x_center, y_center, width, height = boxes.unbind(dim=1)
            x_min = x_center - 0.5 * width
            y_min = y_center - 0.5 * height
            x_max = x_center + 0.5 * width
            y_max = y_center + 0.5 * height
            boxes = torch.stack([x_min, y_min, x_max, y_max], dim=1)

            # Post confidence filtering
            conf_filtered = torch.cat((boxes, scores.unsqueeze(1), class_ids.unsqueeze(1)), 1)

            # Apply NMS
            keep_indices = box_ops.nms(boxes, scores, iou_thres)

            # Post NMS filtering
            batch_selected_predictions.append(conf_filtered[keep_indices])

        return batch_selected_predictions


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

    pout2 = model.postprocessing(out)

    print("Success!")
    print("feedforward took {:.2f} seconds".format(end - start))

    # Check model size
    print("Total parameters: {:.5f}M".format(sum(p.numel() for p in model.parameters()) / 1e6))
    print("Total trainable parameters: {:.5f}M".format(sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6))
    # Add buffer to parameters
    print("Size of the model: {:.5f}MB".format(sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2))
    # prints more model info inherited from BaseModel
    # print(model.__str__())





