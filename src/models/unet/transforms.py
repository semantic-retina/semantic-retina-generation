from typing import Tuple

from PIL import Image
from torchvision import transforms as T


def make_transforms(img_size: int) -> Tuple[T.Compose, T.Compose]:
    image_transform = T.Compose([T.Resize(img_size), T.ToTensor()])
    label_transform = T.Compose([T.Resize(img_size, Image.NEAREST), T.ToTensor()])
    return image_transform, label_transform
