import os
import torchvision.transforms as tfs
import torch
import torch.utils.data
import numpy as np
import h5py
from PIL import Image

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, key, image_size=64, crop=None, is_validation=False, fraction=1):
        super(ImageDataset, self).__init__()
        self.root = data_dir
        self.image_size = image_size
        self.crop = [8, 14, 100, 100]
        self.is_validation = is_validation
        self.data = None
        with h5py.File(self.root, 'r') as f:
            self.data = np.array(f[key])
        self.size = len(self.data)

        fraction = int(fraction * self.size)
        self.data = self.data[:fraction]
        self.size = len(self.data)
        print("Size of dataset: {}".format(self.size))

    def transform(self, img, hflip=False):
        img = Image.fromarray(img)
        img = tfs.functional.resize(img, (self.image_size, self.image_size))
        img = tfs.functional.crop(img, *self.crop)
        if hflip:
            img = tfs.functional.hflip(img)
        return tfs.functional.to_tensor(img)

    def __getitem__(self, index):
        img = self.data[index]
        hflip = not self.is_validation and np.random.rand()>0.5
        return self.transform(img, hflip=hflip)

    def __len__(self):
        return self.size

    def name(self):
        return 'ImageDataset'
