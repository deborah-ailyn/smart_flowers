
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
    model = None
    return model

def classify_image(image, model):
    pred_flower = None
    probability = None
    return probability, pred_flower

def get_pred_img(pred_flower):
    pred_img = None
    return pred_img

def display_result(pred_img, pred_flower, probability):
    pass


if __name__ == "__main__":
    
    st.title("Hello! This is my first steamlit app")

    image = build_image_uploader()

    if image:
        display_image(image)
        success_msg("Image successfully uploaded")
    
    puppies_checkbox()

    # Write code here:

    model = load_model()

    probability, pred_flower = classify_image(image,model)

    predicted_image = get_pred_img(pred_flower)

    display_result(pred_img=predicted_image, pred_flower=pred_flower, probability=probability)

    