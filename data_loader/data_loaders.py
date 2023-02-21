from torchvision import datasets, transforms
from base import BaseDataLoader


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


if __name__ == '__main__':
    #data_load_mnist = MnistDataLoader(data_dir="data/mnist", batch_size=1, shuffle=True, validation_split=0.0, num_workers=1, training=True)
    data_load_voc = VOCDetectionDataLoader(data_dir="./data/VOC2007", batch_size=1, shuffle=True, validation_split=0.0, num_workers=1, training=True)

