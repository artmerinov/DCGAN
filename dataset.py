import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class CelebaDataset(Dataset):
    def __init__(self, folder_path: str, transform, augment=None) -> None:
        self.folder_path = folder_path
        self.transform = transform
        self.augment = augment
        self.image_filenames = os.listdir(folder_path)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img_filename = self.image_filenames[idx]
        img_path = os.path.join(self.folder_path, img_filename)
        img = Image.open(img_path)
        img = self.transform(img)
        if self.augment:
            img = self.augment(img)
        return img

    def __len__(self) -> int:
        return len(self.image_filenames)
    

def transform(img) -> torch.Tensor:
    transform_func = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])
    img = transform_func(img)
    return img


def augment(img) -> torch.Tensor:
    augmentation_func = transforms.Compose([
        transforms.RandomApply([transforms.RandomHorizontalFlip()], p=0.5),
    ])
    img = augmentation_func(img)
    return img


def inverse_normalize(img: torch.Tensor) -> torch.Tensor:
    inv_normalize = transforms.Normalize(mean=(-1.0, -1.0, -1.0), std=(2.0, 2.0, 2.0))
    img = inv_normalize(img)
    img = torch.clamp(img, 0.0, 1.0)
    return img