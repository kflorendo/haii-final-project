from django.shortcuts import render

def home(request):
    return render(request, 'fruitapp/home.html', {})

def about(request):
    return render(request, 'fruitapp/about.html', {})