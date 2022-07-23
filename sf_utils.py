import os
import random
import numpy as np 
from PIL import Image
import streamlit as st
from displayers import display_molly_pic
from machine_learning.clf_training import img_height, img_width


def build_image_uploader():
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file:
        image = Image.open(uploaded_file)      
        return image

def success_msg(msg):
    st.success(msg)
    
def puppies_checkbox():
    result = st.checkbox("Check the box if you like puppies!")

    if result:
        st.write('Great!')        
        display_molly_pic()


def interpret_image(image):
    image = np.array(image)
    image = Image.fromarray(image, mode="RGB")
    image = np.array(image)  
    image = np.resize(image, (img_height, img_width, 3))
    image = np.expand_dims(image, axis=0)
    image = image / 255
    return image

def classify_image(image, model):
    prediction = model.predict(image)
    print(prediction)
    # prediction /= np.sum(np.abs(prediction))
    # prediction[prediction < 0] = 0
    labels = dict(zip([0,1,2,3], ["daisy", "roses", "sunflowers", "tulips"]))
    max_prob = 0
    max_class = None

    for i in range(len(prediction[0])):
        if prediction[0][i] > max_prob:
            max_prob = prediction[0][i]
            max_class = i
    
    probability = max_prob
    predicted_class = labels[max_class]
    probability = int(round(probability, 2) * 100) / 100


    # st.subheader(f"The predicted class is {labels[max_class]} with probability: {probability}")
    success_msg("Sucessful Prediction")
    st.subheader("Predicted class and probability:")
    st.metric(label="", value=labels[max_class].capitalize(), delta=str(probability*100)+"%")
    
    return probability, predicted_class

def get_similar_flowers(pred_flower, n=3):
    path = f"machine_learning/ml_data/flower_photos/{pred_flower}/"
    files = os.listdir(path)
    selected_files = random.sample(files,n)
    images = list(map(lambda f: Image.open(path + f), selected_files))
    return images
    

def display_result(pred_img, pred_flower, probability):
    pass
