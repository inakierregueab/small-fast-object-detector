import torch.nn.functional as F


def nll_loss(output, target):
    return F.nll_loss(output, target)


def yolov5_loss(predictions, targets, num_classes, anchors, device):
    """
    Compute the YOLOv5 loss function for a batch of predictions and targets.

    Args:
        predictions (tensor): Tensor of shape (batch_size, predictions_per_scale, grid_y, grid_x, xywh + c + class_probs)
        targets (tensor): Tensor of shape (batch_size, class_id + xywh)
        num_classes (int): Number of classes in the dataset
        anchors (list): List of anchor boxes for each prediction scale
        device (str): Device to use for computation (e.g. 'cpu', 'cuda')

    Returns:
        Tensor: Scalar tensor representing the total loss for the batch
    """
    loss = 0

    # Loop over each image in the batch
    for i in range(predictions.shape[0]):

        # Separate the predictions and targets for the current image
        pred = predictions[i]
        target = targets[i]

        # Convert the targets to the same format as the predictions
        target = convert_targets(target, anchors, num_classes, device)

        # Loop over each prediction scale
        for j in range(len(pred)):
            # Separate the predictions and targets for the current scale
            pred_scale = pred[j]
            target_scale = target[j]

            # Compute the mask for the positive targets
            mask = target_scale[:, 4] > 0

            # Compute the localization loss
            box_loss = bbox_iou_loss(pred_scale[:, :4], target_scale[:, :4])

            # Compute the confidence loss
            obj_loss = binary_cross_entropy(pred_scale[:, 4], mask.float())
            noobj_loss = binary_cross_entropy(pred_scale[:, 4], 1 - mask.float())
            conf_loss = 0.5 * obj_loss + 0.5 * noobj_loss

            # Compute the classification loss
            class_loss = binary_cross_entropy(pred_scale[:, 5:], target_scale[:, 5:])

            # Compute the total loss for the current scale
            scale_loss = box_loss + conf_loss + class_loss

            # Add the scale loss to the total loss for the image
            loss += scale_loss.sum()

    # Normalize the total loss by the batch size and return
    return loss / predictions.shape[0]
