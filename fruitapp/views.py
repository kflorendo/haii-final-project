from django.shortcuts import render

def home(request):
    return render(request, 'fruitapp/home.html', {})

def about(request):
    return render(request, 'fruitapp/about.html', {})

def classify(request):
    return render(request, 'fruitapp/classify.html', {})

def classify_image(request):
    return render(request, 'fruitapp/classify.html', {})