import streamlit as st
from PIL import Image


def display_image(image):
    st.image(image, caption='Uploaded Image.', use_column_width=True)

def display_molly_pic():
    molly_pic = Image.open('data/molly_pic.jpeg')
    st.image(molly_pic, caption='This is Molly, my puppy',use_column_width=True)
