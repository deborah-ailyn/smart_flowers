
import streamlit as st
from displayers import display_image, display_similar_flowers
from machine_learning.utils import load_model
from sf_utils import build_image_uploader, get_similar_flowers, interpret_image, classify_image, get_similar_flowers, display_result



if __name__ == "__main__":
    
    st.title("Welcome to Smart Flower")
    st.subheader("A quick AI solution to classify flower pictures")

    st.subheader(f"Load a flower picture, our AI will classify it into our categories: [daisy,  roses,  sunflowers,  tulips]")

    image = build_image_uploader()

    if image:
        display_image(image)
        
        test_image = interpret_image(image)
        model = load_model()
        probability, predicted_flower = classify_image(test_image, model)
        show_images = get_similar_flowers(predicted_flower)

        display_similar_flowers(show_images)


    