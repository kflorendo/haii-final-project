# Fruit App: 05-318 Human AI Interaction Final Project

## About
Fruit App is a web application that uses AI to classify a fruit as fresh or rotten. Here are the components of this repo:

### Web application
This web application allows users to upload a picture of their fruit and classify if it is as fresh or rotten. At the home page, if the user is new to the site, they can go to the about page and learn about what Fruit App is, how it works, and what are the stakes of using this app, particularly metrics (accuracy, false positive rate, and false negative rate) of the models. Otherwise, if the user is familiar with the app, they can click a button to start classifying. The user classifies a fruit in 3 steps:
1. Upload: The user can select fruit type, upload a picture, and preview it
2. Results: The user can see the results of classifying the fruit with the model. Here, they can see the confidence level of the model, and if they're still wary of the model's prediction, we provide descriptions of what fresh and rotten fruit of that fruit type look like.
3. Explain: If the user wants to know why the model made that prediction, we use Lime (local interpretable model-agnostic explanations) to show "pros and cons", where pros are areas of the image that support the model's prediction, and cons are areas of the image that go against the model's prediction.

### Classify script
The script `scripts/classify.py` is a proof-of-concept script that classifies an image of a banana, prints the class name to the terminal, and uses LIME to explain the models prediction. The result from Lime is displayed to the user using matplotlib.

### Metrics script
The script `scripts/metrics.py` calculates various metrics for a specified fruit classification model. These metrics include true positive, true negatives, false positives, false negatives, accuracy, false positive rate, and false negative rate. The results are printed to the terminal and written to a text file in the folder `scripts/metrics`.

## Technologies and Machine Learning

### Technologies Used
* Web app: Django, Python, HTML, CSS, JavaScript, Boostrap 5
* Machine learning: [Google Teachable Machine](https://teachablemachine.withgoogle.com/), Tensorflow, Keras, [Lime](https://github.com/marcotcr/lime), skimage, matplotlib, numpy

### Datasets Used
* [Fresh and Rotten Fruits Dataset for Machine-Based Evaluation of Fruit Quality](https://data.mendeley.com/datasets/bdd69gyhv8/1) (Mendeley)
* [Fruits fresh and rotten for classification](https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification) (Kaggle)

### Models
There are 8 models located in the `models` folder for the following fruits: banana, apple, orange, strawberry, mango, tomato, green grape, and lime. The models were trained using [Google Teachable Machine](https://teachablemachine.withgoogle.com/) and saved as Keras models.

### Open Source Code + New Implementation
I referenced this [YouTube tutorial](https://www.youtube.com/watch?v=RvnpVJApBz8) for learning how to set up Tensorflow in a Django app (particularly, loading and using models). I referenced [Tutorial - Image Classification Keras](https://github.com/marcotcr/lime/blob/master/doc/notebooks/Tutorial%20-%20Image%20Classification%20Keras.ipynb) from the LIME GitHub page for learning how to use Lime to identify pros and cons in an image. I referenced this [example from Teachable Machine](https://github.com/googlecreativelab/teachablemachine-community/blob/master/snippets/markdown/image/tensorflow/keras.md) to learn how to load a Keras model, process the image, and make predictions.

I implemented the rest of the code, including:
* the UI with HTML, CSS, and JS web pages that display info and respond to user actions
  * functionality for uploading + previewing images then feeding them to the model
  * displaying classification data to the user
  * writing descriptions and calculating metrics for the about page
* the Django backed
  * setting up urls and actions
  * configuring settings (setting up static files and temporary storage)
* metrics.py and classify.py scripts

## Getting Started

### Running the web application
Here are the steps to running this app locally.
1. Clone the repository.
2. Create a virtual environment with the command `python3 -m venv /path/to/new/venv`.
3. Enter the virtual environement using `source /path/to/new/venv/bin/activate`.
4. Install requirements using `pip install -r requirements.txt`.
5. Set up the Django app:
  * Run `python3 manage.py makemigrations fruitapp`.
  * Run `python3 manage.py migrate`.
6. Run the Django app using `python3 manage.py runserver`. You can view the app in your browser by going to `localhost:8000`

### Running classify.py
From the top level directory, run `python3 scripts/classify.py`.

### Running metrics.py
From the top level directory, run `python3 scripts/metrics.py [fruit name]` where `[fruit name]` is a fruit we have a model for, such as `banana` and `apple`.
