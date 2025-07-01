import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np

st.markdown("<p style='font-size:20px; color:tomato;'>나와 비슷한 연예인 찾기</p>", unsafe_allow_html=True)

file = st.file_uploader('이미지를 업로드해주세요', type=['jpg', 'png'])

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_Model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r", encoding='UTF-8').readlines()

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

if file is None:
    st.markdown("<p style='font-size:15px; color:gray;'>이미지를 업로드하세요</p>", unsafe_allow_html=True)
else:
    # Replace this with the path to your image
    image = Image.open(file).convert("RGB")
    st.image(image, use_container_width=True)
    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # Predicts the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    
    cname = class_names[index]
    cscore = round(prediction[0][index]*100,2)
    
    st.markdown(f"<p style='font-size:20px; color:blue'>당신은 {cname.split( )[1]} 와 {cscore}% 비슷합니다.</p>", unsafe_allow_html=True)
    
    for i in range(len(prediction[0])):
        if i != index:
            class_name = class_names[i]
            confidence_score = round(prediction[0][i]*100,2)        
            ret = f"<p style='font-size:15px; color:gray;'>{class_name.split( )[1]} = {confidence_score}%</p>"        
            st.markdown(ret, unsafe_allow_html=True)