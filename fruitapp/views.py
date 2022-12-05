from django.shortcuts import render
from django.http import HttpResponse
from django.core.files.storage import default_storage
from django.conf import settings


import numpy as np
# from keras.preprocessing.image import img_to_array, load_img
# from tensorflow.python.keras.backend import set_session

def home(request):
    return render(request, 'fruitapp/home.html', {})

def about(request):
    return render(request, 'fruitapp/about.html', {})

def upload(request):
    return render(request, 'fruitapp/upload.html', {})

def classify(request):
    if request.method == 'POST':
        file = request.FILES['imageFile']
        file_name = default_storage.save(file.name, file)
        file_url = default_storage.path(file_name)
    return render(request, 'fruitapp/upload.html', {})