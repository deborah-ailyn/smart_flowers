
# Helper libraries
from cProfile import label
import matplotlib.pyplot as plt
import numpy as np
from os import listdir
from os.path import join
import cv2
import pandas
from scipy.ndimage import rotate
import os
import random
from tensorflow import keras

data_path = "ml_data/flower_photos/"
# size = 128,128
default_model_path = "machine_learning/ml_models/conv_nn_flower_classifier_64_64_10"

def save_model(model, model_path):
    # Guardar configuración JSON en el disco
    json_config = model.to_json()
    with open(f'{model_path}.json', 'w') as json_file:
        json_file.write(json_config)
    # Guardar pesos en el disco
    model.save_weights(f'{model_path}.h5')
    print(f"Model saved at {model_path}")

def load_model(model_path=default_model_path):
    # Recargue el modelo de los 2 archivos que guardamos
    with open(f'{model_path}.json') as json_file:
        json_config = json_file.read()
    model = keras.models.model_from_json(json_config)
    model.load_weights(f'{model_path}.h5')

    return model

def load_training_data(size):
    folders = os.listdir(data_path)
    # Import the images and resize them to a 128*128 size
    # Also generate the corresponding labels

    image_names = []
    train_labels = []
    train_images = []


    for folder in folders:
        for file in os.listdir(os.path.join(data_path,folder)):
            if file.endswith("jpg"):
                image_names.append(os.path.join(data_path,folder,file))
                train_labels.append(folder)
                img = cv2.imread(os.path.join(data_path,folder,file))
                im = cv2.resize(img,size)
                train_images.append(im)
            else:
                continue
    
    predictors = np.array(train_images)

    predictors = predictors.astype('float32') / 255.0

    label_dummies = pandas.get_dummies(train_labels)

    labels =  label_dummies.values.argmax(1)
    # predictors, labels = data_augmentation(predictors, labels)

    # Shuffle the labels and images randomly for better results

    union_list = list(zip(predictors, labels))
    random.shuffle(union_list)
    predictors,labels = zip(*union_list)

    # Convert the shuffled list to numpy array type

    predictors = np.array(predictors)
    labels = np.array(labels)
    return predictors, labels

def data_augmentation(predictors, labels):
    i = 0

    for t, l in zip(predictors, labels):

        predictors = np.concatenate([np.expand_dims(translate(t, direction="up", shift=20), axis=0), predictors], axis=0)
        labels = np.append(labels, l)

        predictors = np.concatenate([np.expand_dims(translate(t, direction="down", shift=20), axis=0), predictors], axis=0)
        labels = np.append(labels, l)

        predictors = np.concatenate([np.expand_dims(translate(t, direction="left", shift=20), axis=0), predictors], axis=0)
        labels = np.append(labels, l)

        predictors = np.concatenate([np.expand_dims(translate(t, direction="right", shift=20), axis=0), predictors], axis=0)
        labels = np.append(labels, l)

        # predictors = np.concatenate([np.expand_dims(random_crop(t), axis=0), predictors], axis=0)
        # labels = np.append(labels, l)

        # predictors = np.concatenate([np.expand_dims(random_crop(t), axis=0), predictors], axis=0)
        # labels = np.append(labels, l)

        # predictors = np.concatenate([np.expand_dims(random_crop(t), axis=0), predictors], axis=0)
        # labels = np.append(labels, l)

        predictors = np.concatenate([np.expand_dims(rotate_img(t, 90), axis=0), predictors], axis=0)
        labels = np.append(labels, l)

        predictors = np.concatenate([np.expand_dims(rotate_img(t, 180), axis=0), predictors], axis=0)
        labels = np.append(labels, l)

        predictors = np.concatenate([np.expand_dims(rotate_img(t, 45), axis=0), predictors], axis=0)
        labels = np.append(labels, l)

        i += 1
        if i > 100:
            break

    return predictors, labels

def random_crop(img, crop_size=(5, 5)):
    assert crop_size[0] <= img.shape[0] and crop_size[1] <= img.shape[1], "Crop size should be less than image size"
    img = img.copy()
    w, h = img.shape[:2]
    x, y = np.random.randint(h-crop_size[0]), np.random.randint(w-crop_size[1])
    img = img[y:y+crop_size[0], x:x+crop_size[1]]
    return img

def rotate_img(img, angle, bg_patch=(5,5)):
    assert len(img.shape) <= 3, "Incorrect image shape"
    rgb = len(img.shape) == 3
    if rgb:
        bg_color = np.mean(img[:bg_patch[0], :bg_patch[1], :], axis=(0,1))
    else:
        bg_color = np.mean(img[:bg_patch[0], :bg_patch[1]])
    img = rotate(img, angle, reshape=False)
    mask = [img <= 0, np.any(img <= 0, axis=-1)][rgb]
    img[mask] = bg_color
    return img

def translate(img, shift=10, direction='right', roll=True):
    assert direction in ['right', 'left', 'down', 'up'], 'Directions should be top|up|left|right'
    img = img.copy()
    if direction == 'right':
        right_slice = img[:, -shift:].copy()
        img[:, shift:] = img[:, :-shift]
        if roll:
            img[:,:shift] = np.fliplr(right_slice)
    if direction == 'left':
        left_slice = img[:, :shift].copy()
        img[:, :-shift] = img[:, shift:]
        if roll:
            img[:, -shift:] = left_slice
    if direction == 'down':
        down_slice = img[-shift:, :].copy()
        img[shift:, :] = img[:-shift,:]
        if roll:
            img[:shift, :] = down_slice
    if direction == 'up':
        upper_slice = img[:shift, :].copy()
        img[:-shift, :] = img[shift:, :]
        if roll:
            img[-shift:,:] = upper_slice
    return img

def gaussian_noise(img, mean=0, sigma=0.03):
    img = img.copy()
    noise = np.random.normal(mean, sigma, img.shape)
    mask_overflow_upper = img+noise >= 1.0
    mask_overflow_lower = img+noise < 0
    noise[mask_overflow_upper] = 1.0
    noise[mask_overflow_lower] = 0
    img += noise
    return img