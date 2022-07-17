import streamlit as st
from PIL import Image
from displayers import display_image, display_molly_pic

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

def load_model():
    pass

def classify_image():
    pass

def display_image():
    pass


if __name__ == "__main__":
    
    st.title("Hello! This is my first steamlit app")

    image = build_image_uploader()

    if image:
        display_image(image)
        success_msg("Image successfully uploaded")
    
    puppies_checkbox()

    load_model()

