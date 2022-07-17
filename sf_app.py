import streamlit as st
from PIL import Image



def build_image_uploader():
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file:
        image = Image.open(uploaded_file)        
        return image


def display_image(image):
    st.image(image, caption='Uploaded Image.', use_column_width=True)

if __name__ == "__main__":
    
    st.title("Hello! This is my first steamlit app")

    image = build_image_uploader()

    if image:
        display_image(image)