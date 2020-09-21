import os
import json
import random

from PIL import Image

import albumentations as A
from albumentations.pytorch import ToTensorV2

import numpy as np

import torch
from torch.utils.data import Dataset

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(0)
np.random.seed(0)


class WaterDataset(Dataset):
    def __init__(self, file_path, transform=None):
        super().__init__()
        with open(file_path, 'r') as f:
            self.image_list = json.load(f)
        self.transforms = transform
        self.to_tensor = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406, 0],
                                std=[0.229, 0.224, 0.225, 1]),
            ToTensorV2()
        ])
        
    def __len__(self):
        # return 100
        return len(self.image_list)
        
    def __getitem__(self, idx):
        file_name = self.image_list[idx]
        
        I = np.asarray(Image.open(file_name)) # h x w x 4
        # print(I.shape, I.dtype)
        if self.transforms is not None:
            I = self.transforms(image=I)['image'] # h x w x 4
        I = self.to_tensor(image=I)['image']
        im = I[:3, :, :]
        gt = I[[-1], :, :]
        return im, gt
    
train_transform = A.Compose([
    A.HorizontalFlip(),
    A.VerticalFlip(),
    A.RandomRotate90(),
    A.RandomResizedCrop(224,224, scale=(0.5, 1.0))
])

test_transform = A.Compose([
    A.Resize(224,224)
])
