import streamlit as st
from PIL import Image
import numpy as np
from displayers import display_image, display_molly_pic
from machine_learning.utils import load_model
from machine_learning.clf_training import img_height, img_width
import os, random
import wikipediaapi


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
    return image

def classify_image(image, model):
    prediction = model.predict(image)
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


    st.subheader("Predicted class and probability:")
    st.metric(label="", value=labels[max_class].capitalize(), delta=str(probability*100)+"%")
    
    return probability, predicted_class

def get_similar_flowers(pred_flower, n=3):
    path = f"machine_learning/ml_data/flower_photos/{pred_flower}/"
    files = os.listdir(path)
    selected_files = random.sample(files,n)
    images = list(map(lambda f: Image.open(path + f), selected_files))
    return images

def show_details_button():
    show_details = st.button('Show Details')
    return show_details

def display_details(predicted_flower):
    page, summary = wikipedia_summary(predicted_flower)
    st.write(summary)
    summary_url(page)

def wikipedia_summary(predicted_flower):
    wiki_wiki = wikipediaapi.Wikipedia('en')
    if predicted_flower == 'daisy':
        page = wiki_wiki.page('Bellis perennis')
        summary = page.summary

    elif predicted_flower == 'roses':
        page = wiki_wiki.page('roses')
        summary = page.summary       
    
    elif predicted_flower == 'sunflowers':
        page = wiki_wiki.page('sunflowers')
        summary = page.summary 

    elif predicted_flower == 'tulips':
        page = wiki_wiki.page('tulips')
        summary = page.summary

    return page, summary    

def summary_url(page):
    url = page.fullurl
    st.write(f'Read more [here]({url})')
