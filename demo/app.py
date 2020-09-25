import streamlit as st
from PIL import Image
from inference import get_model, inference, draw_mask

get_model = st.cache(get_model)

models = {
    'model 1': get_model('cuda:0'),
    'model 2': get_model('cuda:1'),    
}

st.set_option('deprecation.showfileUploaderEncoding', False)
model_id = st.sidebar.radio('Select model', ('model 1', 'model 2'))

'''
# Lake segmentation

wow? it's cool!
'''

option = st.selectbox('Select image', ('Upload your image', '1.jpg', '2.jpg'))

if option == 'Upload your image':
    image = st.file_uploader('select bbox of lake', type=['.jpg', '.png'])
    if image is not None:
        image = np.array(Image.open(img_file_buffer))
else:
    image = Image.open(option)

if image is not None:
    st.image(image, use_column_width=True)

    res = inference(model, image)
    
    collage = draw_mask(image, res)
    
    st.image(collage, use_column_width=True)