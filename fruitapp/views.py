from django.shortcuts import render
from django.http import HttpResponse
from django.core.files.storage import default_storage
from django.conf import settings


import numpy as np
# from keras.preprocessing.image import load_img
from PIL import Image, ImageOps #Install pillow instead of PIL
from tensorflow.python.keras.backend import set_session

def home(request):
    return render(request, 'fruitapp/home.html', {})

def about(request):
    return render(request, 'fruitapp/about.html', {})

def upload(request):
    return render(request, 'fruitapp/upload.html', {})

def classify(request):
    if not request.method == 'POST':
        return render(request, 'fruitapp/upload.html', {})
    
    file = request.FILES['imageFile']
    file_name = default_storage.save(file.name, file)
    file_url = default_storage.path(file_name)

    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1.
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Replace this with the path to your image
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
    with settings.GRAPH1.as_default():
        set_session(settings.SESS)
        predictions = settings.BANANA_MODEL.predict(data)
        index = np.argmax(predictions)
        class_name = settings.FRUIT_CLASS_NAMES[index]
        confidence_score = predictions[0][index]
    return render(request, 'fruitapp/classify.html', {'predictions': predictions, 'class_name': class_name, 'confidence_score': confidence_score, 'file_url': file_url})

def explain(request):
    return render(request, 'fruitapp/explain.html', {})