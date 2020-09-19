import os
import json
import random
random.seed(0)

from PIL import Image

import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from sklearn.model_selection import train_test_split

import torch
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import numpy as np
np.random.seed(0)

from torch.optim import Adam
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm

from livelossplot import PlotLosses

import ternausnet.models

class WaterDataset(Dataset):
    def __init__(self, file_path, transform=None):
        super().__init__()
        with open(file_path, 'r') as f:
            self.image_list = json.load(f)
        self.transforms = transform
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406, 0],
                                std=[0.229, 0.224, 0.225, 1])
        ])
        
    def __len__(self):
        # return 100
        return len(self.image_list)
        
    def __getitem__(self, idx):
        file_name = self.image_list[idx]
        
        I = Image.open(file_name) # h x w x 4
        if self.transforms is not None:
            I = self.transforms(I) # h x w x 4
        I = self.to_tensor(I)
        im = I[:3, :, :]
        gt = I[[-1], :, :]
        return im, gt
    
class RandomRotate90:
    def __init__(self):
        pass
    def __call__(self, img):
        return img.rotate(random.choice([0,90,180,270]))
    
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    RandomRotate90(),
    transforms.RandomResizedCrop(size=(224,224), scale=(0.5, 1.0))
])

test_transform = transforms.Compose([
    transforms.Resize(size=(224,224))
])


def viz(model, path, device):
    I = Image.open(path).resize((224,224), resample=Image.BILINEAR)
    I = np.asarray(I)
    II = I[:, :, :3].copy()
    I = torch.tensor(I, dtype=torch.float)
    I = I / 255
    I = I.transpose(1, 2).transpose(0, 1)
    im = I[:3, :, :]
    
    im = (im - torch.tensor([0.485, 0.456, 0.406])[..., None, None]) / torch.tensor([0.229, 0.224, 0.225])[..., None, None]
    
    gt = (I[3, :, :]).numpy()
    im = im[None, ...]
    im = im.to(device)
    
    pred = model(im).detach().cpu().numpy()[0, 0]
    mask_pred = (pred > 0.5)
    
    gt = gt > 0.5
    
    print(mask_pred.sum())
    print(II.shape, mask_pred.shape)
    II[:, :, 0][mask_pred] = 255
    II[:, :, 1][gt] = 255
    return II
    