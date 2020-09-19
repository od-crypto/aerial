import numpy as np
import requests
from urllib.request import urlopen
#from PIL import Image
import PIL.Image
#from PIL import ImageDraw
import cv2
import os
import demjson
from io import BytesIO
import matplotlib.pyplot as plt
from IPython.display import Image
from IPython.display import display


def create_collage(url, image_json, path_image, path_collage, path_mask, display_locally):
    
    I = np.asarray(PIL.Image.open(BytesIO(requests.get(url).content)))
    
    # print(I.shape)
    
    polys = [create_poly(I.shape[:2], item['data']) for item in image_json]
    mask = ((np.stack(polys, axis=0).sum(axis=0)) % 2).astype(np.uint8)
    boundary = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, np.ones((15,15), dtype=np.uint8))
    II = I.copy()
    II[boundary > 0] = [0,0,255]
    
    # print(mask.shape, mask.dtype)
    
    # cv2 saves in bgr, however Image reads in rgb:
    # cv2.imwrite(path_mask, mask*255)
    # cv2.imwrite(path_collage, II[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 100])
    # cv2.imwrite(path_image, I[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 100])
    PIL.Image.fromarray(mask * 255).save(path_mask)
    PIL.Image.fromarray(II).save(path_collage)
    PIL.Image.fromarray(I).save(path_image)
    
    if display_locally==True:
        plot_locally(url, II, I, mask)


def create_poly(shape, vs):
    vsl = [[item['x'] * shape[1], item['y'] * shape[0]] for item in vs]
    vsnp = np.rint(np.array(vsl)).astype(np.int32)
    II = np.zeros(shape, dtype=np.uint8)
    return cv2.fillPoly(II, [vsnp], 1)


def plot_locally(url, II, I, mask):

    fig, ax2 = plt.subplots(figsize=(15, 10))
    ax2.set_title(url)
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.set_title(url)
    fig, ax_mask = plt.subplots(figsize=(15, 10))
    ax_mask.set_title(url)
    
    ax2.imshow(II, interpolation='nearest')
    ax.imshow(I, interpolation='nearest')
    ax_mask.imshow(mask, interpolation='nearest')

    return plt.tight_layout()


def create_masks(df, image_dir, collage_dir, mask_dir, display_locally=True, init=0, fin=None):
        
    # Create masks, collages and images folders, 
    # if they do not exist:
    
    if not os.path.exists(f"{mask_dir}/"):
        os.makedirs(f"{mask_dir}/")
        
    if not os.path.exists(f"{collage_dir}/"):
        os.makedirs(f"{collage_dir}/")
    
    if not os.path.exists(f"{image_dir}/"):
        os.makedirs(f"{image_dir}/")

    
    fin = fin or len(df)
    print("Create masks for images: ", init + 1, " to ", fin)
    l = [] 
    for i in range(init, fin):
    
        #############################
        # test image with i=121:
        if i!=121:
            # continue
            pass
        #############################
    
        image_url = df.iloc[i,:]['INPUT:image']
        # print(i, image_url)
    
        path_image = f"{image_dir}/{image_url.split('/')[-1][:-4]}.png"
        path_collage = f"{collage_dir}/{image_url.split('/')[-1][:-4]}.png"
        path_mask = f"{mask_dir}/{image_url.split('/')[-1][:-4]}.png"
    
        json_string = df.iloc[i,:]['OUTPUT:result']
        image_json = demjson.decode(json_string)
    
        l.append(image_url)
        
        create_collage(image_url, image_json, path_image, path_collage, path_mask, display_locally)
        
        #############################
        # first four masks:
        # if i >= 3:
            # break
        #############################
        
    return print('Total masks created: ', len(l))






################################################################
# Compare: 
# https://gist.github.com/Kucev/b1929ad363f8c17fdf9b25014e138b10