from tabnanny import verbose
# from utils import load_training_data, save_model
# from model import build_model
import numpy as np 
from .utils import load_training_data, save_model
from .model import build_model

model_epochs = 10
img_height, img_width = 64, 64
model_path = f"ml_models/conv_nn_flower_classifier_{img_width}_{img_height}_{model_epochs}"

if __name__=="__main__":
    predictors, labels = load_training_data((img_height, img_width))
    num_classes = len(np.unique(labels))
    print(num_classes)
    model = build_model(num_classes)
    model.fit(predictors, labels, epochs=model_epochs, verbose=True)

    save_model(model, model_path)