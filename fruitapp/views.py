from django.shortcuts import render
from django.http import HttpResponse
from django.core.files.storage import default_storage
from django.conf import settings


import numpy as np
# from keras.preprocessing.image import load_img
from PIL import Image, ImageOps  # Install pillow instead of PIL
from tensorflow.python.keras.backend import set_session

from skimage.segmentation import mark_boundaries
from skimage.io import imsave

import os

def home(request):
    return render(request, 'fruitapp/home.html', {})

def get_metrics():
    metrics_folder = 'scripts/metrics/'
    metrics = []
    num_fruits = 0
    cumulative_accuracy = 0
    cumulative_fpr = 0
    cumulative_fnr = 0
    for file_name in os.listdir('scripts/metrics/'):
        file_url = os.path.join(metrics_folder, file_name)
        with open(file_url, 'r') as f:
            lines = f.readlines()
            accuracy = float(lines[6][len("accuracy: "):]) * 100
            fpr = float(lines[7][len("false positive rate: "):]) * 100
            fnr = float(lines[8][len("false negative rate: "):]) * 100
            metrics.append({
                'fruit': file_name[:-1 * len("_metrics.txt")],
                'accuracy': round(accuracy, 2),
                'fpr': round(fpr, 2),
                'fnr': round(fnr, 2)
            })
            cumulative_accuracy += accuracy
            cumulative_fpr += fpr
            cumulative_fnr += fnr
        num_fruits += 1
    overall_metric = {
        'accuracy': round(cumulative_accuracy / num_fruits, 2),
        'fpr': round(cumulative_fpr / num_fruits, 2),
        'fnr': round(cumulative_fnr / num_fruits, 2),
    }
    return metrics, overall_metric

def about(request):
    metrics, overall_metric = get_metrics()
    return render(request, 'fruitapp/about.html', {'metrics': metrics, 'overall_metric': overall_metric})


def upload(request):
    return render(request, 'fruitapp/upload.html', {})


def process_image(file_url):
    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1.
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Open image
    image = Image.open(file_url).convert('RGB')

    # resize the image to a 224x224 with the same strategy as in TM2:
    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    return data

def get_model(fruit_type):
    if fruit_type == "banana":
        return settings.BANANA_MODEL
    elif fruit_type == "apple":
        return settings.APPLE_MODEL
    elif fruit_type == "orange":
        return settings.ORANGE_MODEL
    elif fruit_type == "strawberry":
        return settings.STRAWBERRY_MODEL
    elif fruit_type == "mango":
        return settings.MANGO_MODEL
    elif fruit_type == "greengrape":
        return settings.GREENGRAPE_MODEL
    elif fruit_type == "tomato":
        return settings.TOMATO_MODEL
    elif fruit_type == "lime":
        return settings.LIME
    else:
        return settings.BANANA_MODEL

def get_qualities(fruit_type):
    if fruit_type == "banana":
        fresh = [
            "is soft but not squishy",
            "is yellow with few brown spots",
            "holds its shape",
        ]
        rotten = [
            "has a dark brown or black peel",
            "has a fermented, alcohol-like smell",
            "may have leaking fluids",
        ]
    else:
        fresh = []
        rotten = []

    return fresh, rotten

def classify(request):
    if not request.method == 'POST':
        return render(request, 'fruitapp/upload.html', {})

    file = request.FILES['imageFile']
    file_name = default_storage.save(file.name, file)
    file_url = default_storage.path(file_name)

    fruit_type = request.POST.get("fruitType")
    model = get_model(fruit_type)
    data = process_image(file_url)

    # run the inference
    with settings.GRAPH1.as_default():
        set_session(settings.SESS)
        predictions = model.predict(data)
        index = np.argmax(predictions)
        class_name = settings.FRUIT_CLASS_NAMES[index][2:]
        # confidence_score = predictions[0][index]
        fresh_probability = round(predictions[0][0] * 100, 2)
        rotten_probability = round(predictions[0][1] * 100, 2)
    
    fresh_qualities, rotten_qualities = get_qualities(fruit_type)

    context = {
        'class_name': class_name,
        'file_url': file_url,
        'file_name': file_name,
        'fresh_probability': fresh_probability,
        'rotten_probability': rotten_probability, 
        'fruit_type': fruit_type,
        'fresh_qualities': fresh_qualities,
        'rotten_qualities': rotten_qualities,
    }
    return render(request, 'fruitapp/classify.html', context)


def explain(request):
    if not request.method == 'POST':
        return render(request, 'fruitapp/upload.html', {})
    file_url = request.POST.get("fileurl")
    file_name = request.POST.get("filename")
    pro_class_name = request.POST.get("classname")
    con_class_name = "rotten" if pro_class_name == "fresh" else "fresh"

    data = process_image(file_url)
    with settings.GRAPH1.as_default():
        set_session(settings.SESS)
        explanation = settings.EXPLAINER.explain_instance((data[0]).astype(
            'double'), settings.BANANA_MODEL.predict, top_labels=5, hide_color=0, num_samples=1000)
        # temp, mask = explanation.get_image_and_mask(
        #     explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)
        temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=10, hide_rest=False)
        im = mark_boundaries(temp / 2 + 0.5, mask)
        explain_file_name = '/explain/' + file_name
        imsave(settings.MEDIA_ROOT + explain_file_name, im)
    return render(request, 'fruitapp/explain.html', {'explain_file_name': explain_file_name, 'pro_class_name': pro_class_name, 'con_class_name': con_class_name})
