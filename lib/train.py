import os
import json
import random

from PIL import Image

import matplotlib.pyplot as plt

import numpy as np

import torch
from torch.optim import Adam
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from livelossplot import PlotLosses

import ternausnet.models

from .dataset import *
from .metrics import *

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)
random.seed(0)

def trainer(cfg, train_id=None, num_workers=20, device=None):
    
    device = device or 'cuda:0' ##
    train_id = train_id or cfg['train_id']
    use_pretrained_vgg=cfg["use_pretrained_vgg"]
    batch_size=cfg["batch_size"]
    lr=cfg["lr"]
    num_epochs=cfg["num_epochs"]   
   
    model = ternausnet.models.UNet11(pretrained=use_pretrained_vgg)
    
    if cfg['pretrained_model'] is not None:
        model.load_state_dict(torch.load(cfg['pretrained_model']))
    model = model.to(device)

    loss = nn.BCEWithLogitsLoss()
   
    optimizer = Adam(model.parameters(), lr)    

    d_train = WaterDataset(cfg['train_img_list'], train_transform)
    d_val = WaterDataset(cfg['test_img_list'], test_transform)
    
    print(d_val[0][0].shape)
    

    dl_train = DataLoader(d_train, batch_size, shuffle=True, num_workers=num_workers)
    dl_val = DataLoader(d_val, batch_size, shuffle=False, num_workers=num_workers)

        
    metrics = {
        'val_acc': AccuracyMetric(0.5),
        'train_acc': AccuracyMetric(0.5),
        'val_loss': LossMetric(),
        'train_loss': LossMetric(),
        'train_lake_acc': LakeAccuracyMetric(0.5),
        'val_lake_acc': LakeAccuracyMetric(0.5),
        'train_nolake_acc': NoLakeAccuracyMetric(0.5),
        'val_nolake_acc': NoLakeAccuracyMetric(0.5),
    }
    
    groups = {
        'accuracy': ['train_acc', 'val_acc'], 
        'bce-loss': ['train_loss', 'val_loss'], 
        'lake-acc': ['train_lake_acc', 'val_lake_acc'],
        'nolake_acc': ['train_nolake_acc', 'val_nolake_acc'],
    }
    plotlosses = PlotLosses(groups=groups)

    topk_val_losses = {}

    for epoch in range(num_epochs):
        print('train step')
        for name, metric in metrics.items():
            metric.reset()

        model.train()
        for idx, (im, gt) in enumerate(dl_train):
            im = im.to(device)
            gt = gt.to(device)
            optimizer.zero_grad()

            pred = model(im)
            L = loss(pred, gt)
            L.backward()
            assert pred.shape == gt.shape
            metrics['train_acc'].append(pred, gt)
            metrics['train_lake_acc'].append(pred, gt)
            metrics['train_nolake_acc'].append(pred, gt)
            metrics['train_loss'].append(L)
            optimizer.step()
        
        torch.cuda.empty_cache()
        
        model.eval()
        print('eval step')
        for idx, (im, gt) in enumerate(dl_val):
            im = im.to(device)
            gt = gt.to(device)
            pred = model(im)
            L = loss(pred, gt)
            metrics['val_acc'].append(pred, gt)
            metrics['val_lake_acc'].append(pred, gt)
            metrics['val_nolake_acc'].append(pred, gt)
            metrics['val_loss'].append(L)

        torch.cuda.empty_cache()
        
        results = {key: metrics[key].result() for key in metrics}
        plotlosses.update(results)
        plotlosses.send()

        for name, metric in metrics.items():
            metric.history()

            
        history = {key: metrics[key].hist for key in metrics}
        
        
        save_models(model, topk_val_losses, metrics['val_loss'].result(), epoch, train_id, save_num_models=3)
    torch.save(model.state_dict(), 'model-latest.pth')
    
    with open(f'history-{train_id}.json', "w") as write_file:
            json.dump(history, write_file, indent=4)

def save_models(model, topk_val_losses, val_loss, epoch, train_id, save_num_models=3):
    if (len(topk_val_losses) < save_num_models) or (val_loss < max(topk_val_losses.keys())):
            if (len(topk_val_losses) > 0) and (val_loss < max(topk_val_losses.keys())):
                argmin = max(topk_val_losses.keys())
                fname = topk_val_losses[argmin]
                os.remove(fname)
                del topk_val_losses[argmin]
            topk_val_losses[val_loss] = f'models/model-{train_id}-{epoch}.pth'
            torch.save(model.state_dict(), f'models/model-{train_id}-{epoch}.pth')