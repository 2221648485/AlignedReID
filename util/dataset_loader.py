import os
from os import path as osp
import torch
from IPython import embed
from PIL import Image
from torch.utils.data import Dataset

from util import data_manager


def read_image(img_path):
    got_img = False
    if not osp.exists(img_path):
        raise IOError(f"{img_path} is not exist")
    while not  got_img:
        try:
            img = Image.open(img_path).convert("RGB")
            got_img = True
        except IOError:
            raise IOError(f"picture don't read successfully")
        pass
    return img

class ImageDataset(Dataset):
    def __init__(self, dataset, transform = None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, item):
        img_path, pid, camid = self.dataset[item]
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img, pid, camid

    def __len__(self):
        return len(self.dataset)

if __name__ == "__main__":
    dataset = data_manager.init_img_dataset(name="market1501")
    train_loader = ImageDataset(dataset.train)
    embed()