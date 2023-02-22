import os
import torch
import torch.utils.data
import torchvision
from PIL import Image
from pycocotools.coco import COCO


class COCODataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation, transform=None):
        """
        Parameters:
            root: path to the images folder
            annotation: path to the annotation json file
            transform: torchvision.transforms for transforms and tensor conversion
        """
        self.root = root
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transform = transform

        self.target_height = 640
        self.target_width = 640

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        coco_annotation = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert('RGB')

        # Target transformation:
        #   Labels are in COCO format start from 1, but PyTorch CrossEntropyLoss expects labels to start from 0
        #   Desired YOLO format is (class_idx, x_center, y_center, width, height)
        #   Original COCO format for bboxes is (x_min, y_min, width, height)
        #   Use normalized values for bboxes so that they are independent of image size

        num_objs = len(coco_annotation)
        target = []

        # TODO: Parallelize this loop
        for i in range(num_objs):
            class_idx = coco_annotation[i]['category_id'] - 1
            x_center = (2*coco_annotation[i]['bbox'][0] + coco_annotation[i]['bbox'][2]) / (2 + img.width)
            y_center = (2*coco_annotation[i]['bbox'][1] + coco_annotation[i]['bbox'][3]) / (2 + img.height)
            width = coco_annotation[i]['bbox'][2] / img.width
            height = coco_annotation[i]['bbox'][3] / img.height
            target.append([class_idx, x_center, y_center, width, height])

        target = torch.as_tensor(target, dtype=torch.float32)


        # Image transformation:
        #   Convert PIL image to PyTorch tensor
        #   Resize image to target size

        img = torchvision.transforms.ToTensor()(img)
        img = torchvision.transforms.functional.resize(img, (self.target_height, self.target_width))

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.ids)


if __name__ == '__main__':
    dataset = COCODataset(root='/Users/mariateresaalvarez-buhillapuig/Desktop/HuPBA/repos/small-fast-object-detector/data/COCO_dataset/images/val2017',
                          annotation='/Users/mariateresaalvarez-buhillapuig/Desktop/HuPBA/repos/small-fast-object-detector/data/COCO_dataset/annotations/instances_val2017.json')
    img, labels = dataset[0]
