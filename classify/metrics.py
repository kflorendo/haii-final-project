from keras.models import load_model
from PIL import Image, ImageOps #Install pillow instead of PIL
import numpy as np

from keras.preprocessing import image
import matplotlib.pyplot as plt

from lime import lime_image

from skimage.segmentation import mark_boundaries

import os
import sys

if len(sys.argv) < 2:
    raise Exception('Enter fruit name as command line argument')
fruit = sys.argv[1]
print('getting metrics for', fruit, 'model')

model_prefix = 'fruitapp/static/fruitapp/models/'
kaggle_prefix = 'classify/kaggle-dataset/'
kaggle_suffix = '/test'
mendeley_prefix = 'classify/mendeley-dataset/OriginalImage/'
mendeley_suffix = '/test'

def get_kaggle(folder_name):
    return kaggle_prefix + folder_name + kaggle_suffix
def get_mendeley(folder_name):
    return mendeley_prefix + folder_name + mendeley_suffix

if fruit == 'banana':
    model_url = model_prefix + 'banana_model.h5'
    fresh_folders = [get_kaggle('F_Banana'), get_mendeley('FreshBanana')]
    rotten_folders = [get_kaggle('S_Banana'), get_mendeley('RottenBanana')]
elif fruit == 'apple':
    model_url = model_prefix + 'apple_model.h5'
    fresh_folders = [get_mendeley('FreshApple')]
    rotten_folders = [get_mendeley('RottenApple')]
elif fruit == 'orange':
    model_url = model_prefix + 'orange_model.h5'
    fresh_folders = [get_kaggle('F_Orange'), get_mendeley('FreshOrange')]
    rotten_folders = [get_kaggle('S_Orange'), get_mendeley('RottenOrange')]
elif fruit == 'strawberry':
    model_url = model_prefix + 'strawberry_model.h5'
    fresh_folders = [get_kaggle('F_Strawberry'), get_mendeley('FreshStrawberry')]
    rotten_folders = [get_kaggle('S_Strawberry'), get_mendeley('RottenStrawberry')]
elif fruit == 'mango':
    model_url = model_prefix + 'mango_model.h5'
    fresh_folders = [get_kaggle('F_Mango')]
    rotten_folders = [get_kaggle('S_Mango')]
elif fruit == 'tomato':
    model_url = model_prefix + 'tomato_model.h5'
    fresh_folders = [get_kaggle('F_Tomato')]
    rotten_folders = [get_kaggle('S_Tomato')]
elif fruit == "greengrape":
    model_url = model_prefix + 'green_grape_model.h5'
    fresh_folders = [get_mendeley('FreshGrape')]
    rotten_folders = [get_mendeley('RottenGrape')]
elif fruit == "lime":
    model_url = model_prefix + 'lime_model.h5'
    fresh_folders = [get_kaggle('F_Lime')]
    rotten_folders = [get_kaggle('S_Lime')]
else:
    raise Exception('Fruit not found')

# Load the model
model = load_model(model_url, compile=False)

# Load the labels
class_names = open('classify/labels.txt', 'r').readlines()

def classify(file_url):
    print('classifying', file_url)
    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1.
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    image = Image.open(file_url).convert('RGB')
    
    #resize the image to a 224x224 with the same strategy as in TM2:
    #resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    #turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index][2:]
    return class_name

TP = 0
FP = 0
FN = 0
TN = 0

for folder_url in fresh_folders:
    for file_name in os.listdir(folder_url):
        file_url = os.path.join(folder_url, file_name)
        if file_name[-4:] == ".jpg":
            class_name = classify(file_url)
            if class_name == "fresh":
                TP += 1
            else:
                FP += 1

for folder_url in rotten_folders:
    for file_name in os.listdir(folder_url):
        file_url = os.path.join(folder_url, file_name)
        if file_name[-4:] == ".jpg":
            class_name = classify(file_url)
            if class_name == "rotten":
                TN += 1
            else:
                FN += 1

print('TP (fresh and predicted fresh):', TP)
print('FP (fresh and predicted rotten):', FP)
print('FN (rotten and predicted fresh):', FN)
print('TN (rotten and predicted rotten):', TN)

accuracy = (TP + TN) / (TP + TN + FP + FN)
false_positive_rate = FP / (FP + TN)
false_negative_rate = FN / (FN + TP)
print('accuracy:', accuracy)
print('false positive rate:', false_positive_rate)
print('false negative rate', false_negative_rate)