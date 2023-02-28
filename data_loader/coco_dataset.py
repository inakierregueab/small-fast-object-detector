import os
import torch
import torch.utils.data
import albumentations as A
import numpy as np

from PIL import Image
from pycocotools.coco import COCO
from albumentations.pytorch import ToTensorV2

from utils.util import iou_width_height


class COCODataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation, anchors, image_size=640, strides=[8, 16, 32], nc=80, transform=None):
        """
        Reads the COCO dataset, changes bboxes to YOLO format, applies augmentation and builds the target tensors
        Parameters:
            root: path to the images folder
            annotation: path to the annotation json file
            anchors: list of anchors for each scale NOT NORMALIZED
            image_size: size of the image
            strides: strides for each scale
            nc: number of classes
            transform: albumentations transforms
        """
        self.root = root
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transform = transform

        self.target_height = image_size
        self.target_width = image_size
        self.S = [image_size//stride for stride in strides]  # Grid size for each scale
        # normalized to perform iou with bbox
        self.anchors = (torch.tensor(anchors[0] + anchors[1] + anchors[2])/ image_size).reshape(9, 2)
        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors // 3
        self.C = nc
        self.ignore_iou_thresh = 0.5

    def __getitem__(self, index):

        #   Get image and bounding boxes
        bboxes, image = self.get_data(index)

        #   Apply augmentations & resize
        if self.transform is not None:
            augmentations = self.transform(image=image, bboxes=bboxes)
            image = augmentations['image']
            bboxes = augmentations['bboxes']

        #   Prepare target data for loss computation
        #   TODO: move this to loss?
        target = self.match_loss_format(bboxes)

        return image, target

    def match_loss_format(self, bboxes):
        """
        Source: https://github.com/aladdinpersson/Machine-Learning-Collection/tree/dev/ML/Pytorch/object_detection/YOLOv3
        """
        # Create empty tensor for every head/scale matching-ish its output dimensions
        # 6 = p_obj + xywh + class_label; p_obj is 1 if object is present in the cell
        targets = [torch.zeros((self.num_anchors // 3, S, S, 6)) for S in self.S]

        for box in bboxes:
            # For each bbox find the best anchor in every scale in terms of IOU
            iou_anchors = iou_width_height(torch.tensor(box[2:4]), self.anchors)
            anchor_indices = iou_anchors.argsort(descending=True, dim=0)
            x, y, width, height, class_label = box
            has_anchor = [False, False, False]

            # Now its time to populate the target tensors
            for anchor_idx in anchor_indices:
                scale_idx = anchor_idx // self.num_anchors_per_scale    # to which scale does the anchor belong
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale   # which anchor on the scale is it
                S = self.S[scale_idx]   # grid size for the current scale
                i, j = int(S * y), int(S * x)   # cell coordinates for the current scale of the current object

                # If the current anchor hasn't been assigned to any object in this cell yet and the current object
                # hasn't been assigned to any anchor yet in the current scale:
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]

                if not anchor_taken and not has_anchor[scale_idx]:
                    # Mark the anchor as taken in the corresponding scale
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1
                    # Compute the coordinates of the box in the cell's coordinate system x,y € [0,1], w,h in cell units
                    x_cell, y_cell = S * x - j, S * y - i
                    width_cell, height_cell = (width * S, height * S,)
                    box_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell])

                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates    # Insert box coordinates
                    targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label)    # Insert class label
                    has_anchor[scale_idx] = True    # Mark the current scale as having an anchor for the current object

                # If the current anchor hasn't been assigned to any object in this cell yet but other anchors have been,
                # if the matching IOU is greater than the threshold, ignore the current object, no penalization.
                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1

        return tuple(targets)

    def get_data(self, index):
        """
        Reads image and target data from files, modifies target data to match YOLO format
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        coco_annotation = coco.loadAnns(ann_ids)
        path = coco.loadImgs(img_id)[0]['file_name']
        image = Image.open(os.path.join(self.root, path)).convert('RGB')

        #   TARGET TRANSFORMATIONS:
        num_objs = len(coco_annotation)
        bboxes = []
        # TODO: Parallelize this loop
        for i in range(num_objs):
            #   1) Labels are in COCO format start from 1, but PyTorch CrossEntropyLoss expects labels to start from 0
            class_idx = coco_annotation[i]['category_id'] - 1
            #   2) Original COCO format for bboxes is (x_min, y_min, width, height), chage to (x_c, y_c, width, height)
            x_center = (2 * coco_annotation[i]['bbox'][0] + coco_annotation[i]['bbox'][2]) / (2 * image.width)
            y_center = (2 * coco_annotation[i]['bbox'][1] + coco_annotation[i]['bbox'][3]) / (2 * image.height)
            #   3) Normalize bboxes so that they are independent of image size
            width = coco_annotation[i]['bbox'][2] / image.width
            height = coco_annotation[i]['bbox'][3] / image.height
            #   4) Append to list of bboxes, class_idx is last for Albumentations
            bboxes.append([x_center, y_center, width, height, class_idx])

        image = np.array(image)
        return bboxes, image

    def __len__(self):
        return len(self.ids)


if __name__ == '__main__':

    anchors = [
        [(10, 13), (16, 30), (33, 23)],  # P3/8
        [(30, 61), (62, 45), (59, 119)],  # P4/16
        [(116, 90), (156, 198), (373, 326)]  # P5/32
        ]

    scale = 1.0
    IMAGE_SIZE = 640
    train_transforms = A.Compose([A.LongestMaxSize(max_size=int(IMAGE_SIZE * scale)), ToTensorV2()],
                                 bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[]))

    dataset = COCODataset(root='/Users/mariateresaalvarez-buhillapuig/Desktop/HuPBA/repos/small-fast-object-detector/data/COCO_dataset/images/val2017',
                          annotation='/Users/mariateresaalvarez-buhillapuig/Desktop/HuPBA/repos/small-fast-object-detector/data/COCO_dataset/annotations/instances_val2017.json',
                          anchors=anchors, transform=train_transforms)
    img, labels = dataset[2]
    x=0
