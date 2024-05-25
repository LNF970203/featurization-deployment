import tensorflow as tf
from keras.layers import GlobalAveragePooling2D
from keras.models import Model


INPUT_SHAPE = (224, 224, 3)


def load_featurized_model():
    """
    Create the featurization model with base model + GlobalAveragePooling2D
    """
    base_model = tf.keras.applications.ConvNeXtXLarge(
        input_shape = INPUT_SHAPE,
        include_top = False,
        weights = './Models/convnext_xlarge_notop.h5'
    )

    # add average pooling to base
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    model_frozen = Model(inputs = base_model.input, outputs = x)

    return model_frozen