import torch
import albumentations as A

from torchvision import datasets, transforms
from base import BaseDataLoader
from data_loader.coco_dataset import COCODataset
from albumentations.pytorch import ToTensorV2


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


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
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.1, num_workers=1, training=True):
        self.data_dir = data_dir

        anchors = [[(10, 13), (16, 30), (33, 23)],  # P3/8
                [(30, 61), (62, 45), (59, 119)],  # P4/16
                [(116, 90), (156, 198), (373, 326)]  # P5/32
                ]

        train_transforms = A.Compose([A.Resize(height=640, width=640),
                                      A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255),
                                      ToTensorV2()],
                                     bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[]))

        self.dataset = COCODataset(root=self.data_dir + '/images/val2017',
                                   annotation=self.data_dir + "/annotations/instances_val2017.json",
                                   anchors=anchors, transform=train_transforms)

        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


if __name__ == '__main__':
    # data_load_mnist = MnistDataLoader(data_dir="data/mnist", batch_size=1, shuffle=True, validation_split=0.0, num_workers=1, training=True)
    # data_load_voc = VOCDetectionDataLoader(data_dir="./data/VOC2007", batch_size=1, shuffle=True, validation_split=0.0, num_workers=1, training=True)
    data_load_coco = COCODataLoader(data_dir="/Users/mariateresaalvarez-buhillapuig/Desktop/HuPBA/repos/small-fast-object-detector/data/COCO_dataset", batch_size=1, shuffle=True, validation_split=0.0,
                                      num_workers=1, training=True)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    train_features, train_labels = next(iter(data_load_coco))
    x=0

