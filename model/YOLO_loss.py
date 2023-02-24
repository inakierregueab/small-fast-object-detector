import torch
import torch.nn as nn

class YOLOLoss(nn.Module):
    def __init__(self, model):
        super(YOLOLoss, self).__init__()
        self.model = model

        self.head_loss = HeadLoss(model)

    def forward(self, outputs, targets):
        """
        Parameters:
            outputs (list): list of tensors of shape (batch_size, num_anchors, grid_size, grid_size, num_classes + 5),
                            one for each head and raw output.
            targets (list):
        """



        pass


class HeadLoss(nn.Module):
    def __init__(self, model):
        super(HeadLoss, self).__init__()
        self.model = model

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

    def forward(self, output, target, anchors):
        pass




