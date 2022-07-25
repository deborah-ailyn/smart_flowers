
from matplotlib.pyplot import show
import streamlit as st
from displayers import display_image, display_similar_flowers
from machine_learning.utils import load_model
from machine_learning.clf_training import img_height, img_width
from sf_utils_v1 import build_image_uploader, display_image, interpret_image, load_model, success_msg, classify_image, get_similar_flowers, show_details_button, display_details
import webbrowser


if __name__ == "__main__":
    
    st.title("Welcome to Smart Flowers")
    st.subheader("A quick AI solution to classify flower pictures")

    st.subheader("Load a flower image. Our AI will classify it according to our categories: [daisy, rose, sunflower, tulip]")

    image = build_image_uploader()

    if image:
        display_image(image)
        success_msg("Image successfully uploaded")
        
        test_image = interpret_image(image)
        model = load_model()
        probability, predicted_flower = classify_image(test_image,model)
        similar_flowers_images = get_similar_flowers(predicted_flower)

        display_similar_flowers(similar_flowers_images)
    
        button = show_details_button()
        if button:
            display_details(predicted_flower)
            

            