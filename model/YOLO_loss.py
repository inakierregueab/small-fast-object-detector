import torch
import torch.nn as nn

from utils import intersection_over_union


class YOLOLoss(nn.Module):
    def __init__(self, anchors):
        super(YOLOLoss, self).__init__()

        # TODO: normalize anchors and express them in terms of the grid size?
        self.anchors = anchors
        self.num_heads = 3

        self.head_loss = HeadLoss()

    def forward(self, outputs, targets):
        loss = 0
        for i in range(3):
            loss += self.head_loss(outputs[i], targets[i], self.anchors[i])

        return loss


class HeadLoss(nn.Module):
    def __init__(self):
        super(HeadLoss, self).__init__()

        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.entropy = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()

        # Constants signifying how much to pay for each respective part of the loss
        # TODO: Should depend on hyperparameters
        self.lambda_class = 1
        self.lambda_noobj = 10
        self.lambda_obj = 1
        self.lambda_box = 10

    def forward(self, predictions, target, anchors):
        # Check where obj and noobj (we ignore if target == -1)
        obj = target[..., 0] == 1  # in paper this is Iobj_i
        noobj = target[..., 0] == 0  # in paper this is Inoobj_i

        # post-processing of predictions
        pred_xy = (predictions[..., 1:3].sigmoid() * 2) - 0.5
        # TODO: anchors normalized?
        pred_wh = ((predictions[..., 3:5].sigmoid() * 2) ** 2) * anchors # cell coords
        pred_box = torch.cat([pred_xy, pred_wh], dim=-1)

        # NO OBJECT LOSS
        no_object_loss = self.bce((predictions[..., 0:1][noobj]), (target[..., 0:1][noobj]))

        # OBJECT LOSS: instead of simple BCE, we use a more complex loss that takes into account the IOU between the
        # target and the prediction anchors for the particular scale in the objectness score so that we can perform MSE
        anchors = anchors.reshape(1, 3, 1, 1, 2)
        ious = intersection_over_union(pred_box[obj], target[..., 1:5][obj]).detach()
        object_loss = self.mse(self.sigmoid(predictions[..., 0:1][obj]), ious * target[..., 0:1][obj])

        # BOX LOSS
        # TODO: use iou loss instead of mse?
        box_loss = self.mse(pred_box[obj], target[..., 1:5][obj])

        # CLASS LOSS
        class_loss = self.entropy((predictions[..., 5:][obj]), (target[..., 5][obj].long()))

        return (
                self.lambda_box * box_loss
                + self.lambda_obj * object_loss
                + self.lambda_noobj * no_object_loss
                + self.lambda_class * class_loss
        )




