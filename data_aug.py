import albumentations as A
from albumentations.pytorch import ToTensorV2

# TODO: params from config file
train_transforms = A.Compose(
    [
        A.Resize(height=640, width=640),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255),
        ToTensorV2()
    ],
    bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[])
)