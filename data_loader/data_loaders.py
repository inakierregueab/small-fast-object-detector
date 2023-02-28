import torch
import albumentations as A
import data_aug

from torchvision import datasets, transforms
from base import BaseDataLoader
from data_loader.coco_dataset import COCODataset
from albumentations.pytorch import ToTensorV2


class VOCDetectionDataLoader(BaseDataLoader):
    """
    VOC object detection data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        self.data_dir = data_dir
        self.dataset = datasets.VOCDetection(self.data_dir, year="2007", image_set='train', download=True)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class COCODataLoader(BaseDataLoader):
    """
    COCO data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, anchors, batch_size, image_size, stride, nc, shuffle=True, validation_split=0.1,
                 num_workers=1, training=True):
        self.data_dir = data_dir
        self.dataset = COCODataset(root=self.data_dir + '/images/val2017',
                                   annotation=self.data_dir + "/annotations/instances_val2017.json",
                                   anchors=anchors, transform=data_aug.train_transforms, image_size=image_size,
                                   stride=stride, nc=nc)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


if __name__ == '__main__':
    # data_load_mnist = MnistDataLoader(data_dir="data/mnist", batch_size=1, shuffle=True, validation_split=0.0, num_workers=1, training=True)
    # data_load_voc = VOCDetectionDataLoader(data_dir="./data/VOC2007", batch_size=1, shuffle=True, validation_split=0.0, num_workers=1, training=True)

    data_load_coco = COCODataLoader(data_dir="/Users/mariateresaalvarez-buhillapuig/Desktop/HuPBA/repos/small-fast-object-detector/data/COCO_dataset", batch_size=1, shuffle=True, validation_split=0.0,
                                      num_workers=1, training=True)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    train_features, train_labels = next(iter(data_load_coco))
    x=0

