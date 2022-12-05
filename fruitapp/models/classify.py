from keras.models import load_model
from PIL import Image, ImageOps #Install pillow instead of PIL
import numpy as np

from keras.preprocessing import image
import tensorflow as tf
import matplotlib.pyplot as plt

from lime import lime_image

from skimage.segmentation import mark_boundaries

g1 = tf.Graph()
sess1 = tf.Session(graph=g1)

with sess1.as_default():
    with g1.as_default():
        tf.global_variables_initializer().run()
        model = load_model('banana_keras_Model.h5', compile=False)

def classify_image(request):
    # Load the model
    model = load_model('banana_keras_Model.h5', compile=False)

    # Load the labels
    class_names = open('models/labels.txt', 'r').readlines()

    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1.
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Replace this with the path to your image
    image = Image.open('static/models/757.jpg').convert('RGB')

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
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    response_data = {'class': class_name, 'confidence_score': confidence_score}
    # json_response = json.dumps(response_data)

    print('Class:', class_name, end='')
    print('Confidence score:', confidence_score)
    return response_data
    # return render(request, 'fruitapp/classify.html', {})