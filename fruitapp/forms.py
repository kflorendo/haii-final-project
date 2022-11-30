from django import forms

class ClassifyFruitForm(forms.Form):
    image = forms.ImageField()