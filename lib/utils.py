from PIL import Image

import numpy as npo
import torch

def viz(model, path, device):
    print("path: ", path)
    I = Image.open(path).resize((224,224), resample=Image.BILINEAR)
    # I = Image.open(path)
    print(I.shape)
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
    