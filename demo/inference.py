import cv2
import ternausnet.models
import numpy as np 
import torch
from PIL import Image
import numpy as np

def get_model(fname=None, device='cuda:0'):
    model = ternausnet.models.UNet11(pretrained=False)
    if fname is not None:
        model.load_state_dict(torch.load(fname, map_location=torch.device('cpu')))
    model = model.to(device=device)
    model.eval()
    return model


def inference(model, I, batch_size=70, threshold=0.5):
    L1, L2 = I.shape[:2]
    d1 = 300
    d2 = 150

    crops = []

    for k, K in zip(list(range(0, L1 - d2, d2)), list(range(d1, L1 - d2, d2)) + [L1]):
        for p, P in zip(list(range(0, L2 - d2, d2)), list(range(d1, L2 - d2, d2)) + [L2]):
            crops.append(I[k:K, p:P])

    batch = np.stack(list(map(lambda x: cv2.resize(x, (224, 224), interpolation=cv2.INTER_AREA), crops)))
    
    batch = ((batch / 255) - np.array([0.485, 0.456, 0.406])[None, None, None, :]) / np.array([0.229, 0.224, 0.225])[None, None, None, :]
    
    batch = torch.tensor(batch, dtype=torch.float).to(device=next(model.encoder.parameters()).device)
    
    batch = batch.transpose(2,3).transpose(1,2)
    
    res = np.concatenate(list(map(lambda x: model(x).detach().cpu().numpy(), torch.split(batch, batch_size))))
    
    C = np.zeros_like(I[:, :, 0])
    II = np.zeros_like(I[:, :, 0], dtype=np.float)
    
    i = 0
    for k, K in zip(list(range(0, L1 - d2, d2)), list(range(d1, L1 - d2, d2)) + [L1]):
        for p, P in zip(list(range(0, L2 - d2, d2)), list(range(d1, L2 - d2, d2)) + [L2]):
            res_ = cv2.resize(res[i, 0], crops[i].shape[:2][::-1], interpolation=cv2.INTER_LINEAR)
            II[k:K, p:P] += res_
            C[k:K, p:P] += 1
            i += 1
    II =  II / C
    
    II = (II > 0.5).astype(np.uint8) * 255
    
    return II

def draw_mask(I, mask, alpha=0.1):
    dup = np.stack([np.zeros_like(mask), np.zeros_like(mask), mask], axis=-1)
    II = (alpha * dup + (1 - alpha) * I).astype(np.uint8)
    brd = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, np.ones((10,10), dtype=np.uint8))
    II[brd > 0] = (0,255,255)   
    
    return II
