import streamlit as st
from PIL import Image


def display_image(image):
    st.subheader("Your image:")
    st.image(image, caption='Uploaded Image.', use_column_width=True)

def display_molly_pic():
    molly_pic = Image.open('data/molly_pic.jpeg')
    st.image(molly_pic, caption='This is Molly, my puppy',use_column_width=True)

def display_similar_flowers(show_images):
    st.subheader("Similar flowers:")
    col1, col2, col3 = st.columns([0.2, 0.2, 0.2])

    col1.image(show_images[0], use_column_width=True)
    col2.image(show_images[1], use_column_width=True)
    col3.image(show_images[2], use_column_width=True)