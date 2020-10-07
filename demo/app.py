import shutil
import os.path
import json

import requests

from PIL import Image
import numpy as np
import pandas as pd

import streamlit as st
st.set_option('deprecation.showfileUploaderEncoding', False)

from inference import get_model, inference, draw_mask

get_model = st.cache(get_model)

########################
## Set models
########################

def download_file(url, fname):
    with requests.get(url, stream=True) as r:
        with open(fname, 'wb') as f:
            shutil.copyfileobj(r.raw, f)

models_config = {
    'model from scratch 19': {
        'device': os.getenv('MODEL1DEVICE', 'cpu'),
        'fname': 'model-from_scratch_total_nofreeze_bs80_ne20_lr0.0001-19.zip',
        'url': 'https://getfile.dokpub.com/yandex/get/https://yadi.sk/d/l2ioWDXYbb9SMw',
    },
    'model vgg sentinel 16': {
        'device': os.getenv('MODEL2DEVICE', 'cpu'),
        'fname': 'model-sentinel_vgg_nofreeze_bs80_ne20_lr0.0001-16.zip',
        'url': 'https://getfile.dokpub.com/yandex/get/https://yadi.sk/d/Yo1uIagOTPBGqA',
    },
}

for model in models_config.values():
    if not os.path.exists(model['fname']):
        download_file(model['url'], model['fname'])
    del model['url']

models = {name: get_model(**models_config[name]) for name in models_config}
##############################



model_id = st.sidebar.radio('Select model', ('model 1', 'model 2'))

'''
# Lake segmentation

wow? it's cool!
'''

option = st.selectbox('Select image', ('Upload your image', 'demo.jpg'))

if option == 'Upload your image':
    image = st.file_uploader('select bbox of lake', type=['jpg', 'png', 'jpeg'])
    if image is not None:
        image = np.asarray(Image.open(img_file_buffer))
else:
    image = np.asarray(Image.open(option))

if image is not None:
    st.image(image, use_column_width=True)

    res = inference(models[model_id], image, batch_size=5)
    
    collage = draw_mask(image, res)
    
    st.image(collage, use_column_width=True)


metrics = {}
with open('history-sentinel_vgg_nofreeze_bs80_ne20_lr0.0001.json', 'r') as f:
    metrics['model vgg sentinel 16'] = pd.DataFrame(json.load(f))
with open('history-from_scratch_total_nofreeze_bs80_ne20_lr0.0001.json', 'r') as f:
    metrics['model from scratch 19'] = pd.DataFrame(json.load(f))

for df in metrics.values():
    for col in df.columns:
        if col.split('_')[0] == 'train':
            df[col] /= ((74774 + 80) // 80)
        else:
            df[col] /= ((1354 + 80) // 80)

ctrl = [
    ('Train accuracy', 'train_acc'),
    ('Val accuracy', 'val_acc'),
    ('Train loss', 'train_loss'),
    ('Val loss', 'val_loss'),
    ('Train lake accuracy', 'train_lake_acc'),
    ('Val lake accuracy', 'val_lake_acc'),
    ('Train nolake accuracy', 'train_lake_acc'),
    ('Val nolake accuracy', 'val_nolake_acc'),
]


for name, metric_name in ctrl:
    st.markdown(f'### {name}')
    st.line_chart({
        'vgg': metrics['model vgg sentinel 16'][metric_name],
        'scratch': metrics['model from scratch 19'][metric_name]
    })