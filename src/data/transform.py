from torch import Tensor
from torchvision.transforms import functional as F


class CropShortEdge:
    def __call__(self, img: Tensor) -> Tensor:
        h, w = img.size
        size = min(h, w)
        img = F.center_crop(img, size)
        return img


class CropLongEdge:
    def __call__(self, img: Tensor) -> Tensor:
        h, w = img.size
        size = max(h, w)
        img = F.center_crop(img, size)
        return img
