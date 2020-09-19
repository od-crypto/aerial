from multiprocessing import Pool
from collections import Counter, defaultdict

import os
import shutil

from matplotlib import pyplot as plt
from PIL import Image
from tqdm import tqdm

import scipy.ndimage
import cv2
import tifffile
import numpy as np



def get_rotated_crops(I, angle, crop_size=(448,448), num_crops=2):
    if angle == 0:
        mask = np.zeros_like(I[:, :, 0])
        mask[:mask.shape[0]-crop_size[0], :mask.shape[1]-crop_size[1]] = 1
        II = I
    else:
        angler = angle * np.pi / 180
        II = scipy.ndimage.rotate(I, angle)
        mask = scipy.ndimage.rotate(np.ones_like(I[:, :, 0]), angle)
        kernel = np.ones((crop_size[0] + 1, crop_size[1] + 1), dtype=np.uint8)
        mask = cv2.erode(mask, kernel, anchor=(0,0))
    
    x, y = mask.nonzero()
    L = len(x)
    for c in np.random.randint(0, L, size=num_crops):
        x0 = x[c]
        y0 = y[c]
        yield II[x0:x0 + crop_size[0], y0:y0 + crop_size[1]]

class DatasetPreprocessor:
    def __init__(self, config):
        for key, value in config.items():
            setattr(self, key, value)
        
        self.fids = self.get_fids()
        
        self._workdirs = (
            self.intermediate, 
            self.rotated_crops, 
            self.rotated_crops_images,
            self.rotated_crops_masks)
        
        for dir_ in self._workdirs:
            os.makedirs(dir_, exist_ok=True)
    
        self.crop_size = (448, 448)
        
    def rmworkdirs(self):
        for dir_ in self._workdirs:
            shutil.rmtree(dir_)
    
    def get_fids(self):
        raise NotImplementedError
        
    def imread(self, fid):
        raise NotImplementedError
        
    def process_mask(self, M):
        '''
        return np.uint8 1-channel binarized 0-255 mask
        '''
        raise NotImplementedError
        
    def rotate_onefile(self, fid, angles=None, num_crops=1000):
        angles = angles or list(range(0, 90, 10))
        I = self.imread(fid)
        
        for angle in angles:
            print(fid, angle)
            for i, patch in enumerate(get_rotated_crops(I, angle, crop_size=self.crop_size, num_crops=num_crops)):
                if patch[:, :, -1].max() == 0 or patch[:, :, -1].min() == 255:
                    continue
                Image.fromarray(patch).save(f'{self.intermediate}/{fid}-{i}-{angle}.png')
    
    def rotate_step(self):
        with Pool(20) as p:
            p.map(self.rotate_onefile, self.fids)

            
    def _ifid_check(self, fname):
        if not fname.endswith('.png'):
            return None
                
        basename = fname.split('.')[0]
        I = self.ifimread(basename)
        if I is None:
            return None
                
        if I.shape != self.crop_size + (4, ):
            return None
            
        return basename

    
    def set_ifids(self):
        with Pool(20) as p:
            ifids = p.map(self._ifid_check, os.listdir(self.intermediate))
            
        ifids = list(filter(lambda x: x is not None, ifids))
        self.ifids = ifids
        
    def _calc_stats(self, ifid):
        I = self.ifimread(ifid)
        return I[:, :, -1].mean() // (0.1 * 255), ifid
    
    def distribution(self):
        with Pool(20) as p:
            distrib = p.map(self._calc_stats, self.ifids)
        return distrib, Counter(map(lambda x: x[0], distrib))
        
    def plot_distrib(self, c):
        plt.bar(c.keys(), c.values())
    
    def ifimread(self, ifid):
        return cv2.imread(f'{self.intermediate}/{ifid}.png', cv2.IMREAD_UNCHANGED)
    
    def set_distrib_max(self, k):
        self.distrib_max = k
        
    def finimwrite(self, ifid):
        I = self.ifimread(ifid)

        cv2.imwrite(f'{self.rotated_crops}/{ifid}.png',I)
        cv2.imwrite(f'{self.rotated_crops_masks}/{ifid}.png',I[:, :, 3])
        cv2.imwrite(f'{self.rotated_crops_images}/{ifid}.png',I[:, :, :3])
        

    def create_balanced_dataset(self, distrib):
        use_fnames = defaultdict(list)
        for mean, fname in filter(lambda x: x is not None, distrib):
            if len(use_fnames[mean]) < self.distrib_max:        
                use_fnames[mean].append(fname)
        use_fnames = [fname for mean in use_fnames 
                      for fname in use_fnames[mean]]
        
        with Pool(20) as p:
            p.map(self.finimwrite, use_fnames)      
        

class LandcoverDatasetPreprocessor(DatasetPreprocessor):
    
    def get_fids(self):
        fids = os.listdir(self.mask_dir)
        _fids = []
        for item in fids:
            for ext in self.allowed_extensions:
                if item.endswith(ext):
                    basename = item.split('.')[0]
                    _fids.append(basename)
        return _fids
        
    
    def imread(self, fid):
        M = tifffile.imread(f'{self.mask_dir}/{fid}.tif')
        I = tifffile.imread(f'{self.image_dir}/{fid}.tif')
        M = self.process_mask(M)
        I = np.concatenate([I, M[..., None]], axis=-1)
        return I
    
    def process_mask(self, M):
        M = M.copy()
        M[M != 3] = 0
        M[M == 3] = 255
        return M
    

class SentinelDatasetPreprocessor(DatasetPreprocessor):
    
    def get_fids(self):
        fids = os.listdir(self.mask_dir)
        _fids = []
        for item in fids:
            for ext in self.allowed_extensions:
                if item.endswith(ext):
                    basename = item.split('.')[0]
                    _fids.append(basename)
        return _fids
        
    
    def imread(self, fid):
        M = cv2.imread(f'{self.mask_dir}/{fid}.png', cv2.IMREAD_UNCHANGED)
        I = cv2.imread(f'{self.image_dir}/{fid}.png', cv2.IMREAD_UNCHANGED)
        print(I.shape)
        M = self.process_mask(M)     
        print(M.shape)
        I = np.concatenate([I, M[..., None]], axis=-1)
        return I
    
    def process_mask(self, M):
        M = M.copy()
        return M