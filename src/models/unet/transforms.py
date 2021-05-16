from typing import Tuple

from torchvision import transforms as T

# Note that the testing and training transformations are the same for the U-Net model.
from torchvision.transforms import InterpolationMode

from src.transforms import joint


def make_transforms(img_size: int) -> Tuple[T.Compose, T.Compose, joint.Compose]:
    image_transform = T.Compose([T.Resize(img_size), T.ToTensor()])
    label_transform = T.Compose(
        [T.Resize(img_size, InterpolationMode.NEAREST), T.ToTensor()]
    )
    joint_transform = joint.Compose(
        [joint.RandomHorizontalFlip(), joint.RandomVerticalFlip()]
    )
    return image_transform, label_transform, joint_transform
