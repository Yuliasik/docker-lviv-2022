import numpy as np
from PIL import Image
from ai.model import get_model, NeuralNetConfig
from os import path

module_dir = path.dirname(path.realpath(__file__))
config = NeuralNetConfig()
model = get_model(config)
model.load_weights(path.join(module_dir, '..', 'weights', 'trained'))

def pixels_to_greyscale(data):
    if len(data) > 0:
        pixel = data[0]
        try:
            channels = len(pixel)
        except TypeError:
            # has no attribute __len__, therefore no tuple, so only greyscale
            channels = 1
    else:
        return data
    if channels == 1:
        return data
    for index, pixel in enumerate(data):
        data[index] = ((pixel[0] + pixel[1] + pixel[2]) / 3) * pixel[4] if channels == 4 else 1
    return data

def image_to_ndarray(image):
    pixels = list(image.getdata())
    pixels = pixels_to_greyscale(pixels)

    # Assuming pixels is a flattened 8x8 image, reshape it to (8, 8, 1)
    pixels = np.asarray(pixels).reshape(8, 8, 1)

    return pixels


def classify(image_file):
    image = Image.open(image_file)

    # convert the image to a numpy array
    pixels = image_to_ndarray(image)

    # add a dimension, which is the batch size dimension (therefore it is 1 for our single sample)
    pixels.shape = (1,) + pixels.shape

    # predict the class (which digit?) for this sample
    # model.predict returns a prediction for each input sample, as list
    # the first one corresponds to the first sample
    # here: each prediction is a vector, with [number of digits: 0-9 = 10] entries
    # each entry is a probability of 0 (0%) - 1 (100%), how likely the input sample is this digit
    prediction = model.predict(pixels)
    return {
        "class": str(prediction[0].argmax())
    }