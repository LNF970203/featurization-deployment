import base64
from io import BytesIO
from PIL import Image
import copy
import numpy as np
from keras.applications.resnet import preprocess_input


IMG_SIZE = (224, 224)

def get_image_payload(data):
    """
    Get the base64 encoded data of an image and
    convert it to a bytearray
    """
    image = base64.b64decode(data)
    payload = bytearray(image)
    return payload


def preprocess_input(img):
    """
    Convert the bytestream to RGB.
    And apply the same preprocessing used in training.
    """
    # convert byte stream to RGB
    stream = BytesIO(img)
    img_array = Image.open(stream).convert("RGB")
    # resize the image
    img_array = img_array.resize(IMG_SIZE)
    # expand the dimension to represent batch
    img_batch = np.expand_dims(img_array, axis = 0)
    # create a deep copy
    image_batch = copy.deepcopy(img_batch)
    preprocess_image_batch = preprocess_input(image_batch)

    return preprocess_image_batch