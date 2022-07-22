
import streamlit as st
from PIL import Image
import numpy as np
from displayers import display_image, display_molly_pic
from machine_learning.utils import load_model
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

<<<<<<< HEAD

def interpret_image(image):
    image = np.array(image)
    image = Image.fromarray(image, mode="RGB")
    image = np.array(image)  
    image = np.resize(image, (img_height, img_width, 3))
    image = np.expand_dims(image, axis=0)
    return image

def classify_image(image, model):
    prediction = model.predict(image)
    prediction /= np.sum(np.abs(prediction))
    prediction[prediction < 0] = 0
    labels = dict(zip([0,1,2,3], ["daisy", "roses", "sunflowers", "tulip"]))
    max_prob = 0
    max_class = None

    for i in range(len(prediction[0])):
        print(prediction[0][i])
        if prediction[0][i] > max_prob:
            max_prob = prediction[0][i]
            max_class = i
    
    probability = max_prob
    predicted_class = labels[max_class]
    print(f"The predicted class is {labels[max_class]} with probability: {max_prob}")
    
    return probability, predicted_class

def get_pred_img(pred_flower):
    pred_img = None
    return pred_img

=======
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

>>>>>>> 94e9419c963fef55c7b8f2865954eefe3f4a9580
def display_result(pred_img, pred_flower, probability):
    pass


if __name__ == "__main__":
    
    st.title("Hello! This is my first steamlit app")

    image = build_image_uploader()

    if image:
        display_image(image)
        success_msg("Image successfully uploaded")
        
        test_image = interpret_image(image)
        model = load_model()
        # probability, pred_flower = classify_image(image,model)
        
        #TODO complete this logical path:

    puppies_checkbox()

    # Write code here:

    model = load_model()

    probability, pred_flower = classify_image(image,model)

    predicted_image = get_pred_img(pred_flower)

    display_result(pred_img=predicted_image, pred_flower=pred_flower, probability=probability)

    