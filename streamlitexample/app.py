import streamlit as st
import numpy as np
from PIL import Image

st.set_option('deprecation.showfileUploaderEncoding', False)

'''
# Olga's great CNN

Let's segment *lakes*!
'''

img_file_buffer = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if img_file_buffer is not None:
    image = np.array(Image.open(img_file_buffer))

    st.image(image, use_column_width=True)

