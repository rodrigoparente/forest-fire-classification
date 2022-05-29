# python imports
import os

# hiding tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# third-party imports
import numpy as np
from PIL import Image

import tensorflow as tf


LABELS = {
    0: 'With Fire',
    1: 'Without Fire' 
}

def main():

    model = tf.keras.models.load_model('./model.h5')

    image = Image.open('./no-fire.jpg')
    image = image.resize((224, 224))

    image = image - np.mean(image)

    input_tensor = np.array(np.expand_dims(image, axis=0), dtype=np.float32)

    output_data = model.predict(x=input_tensor, verbose=0)

    output_data = np.squeeze(output_data)
    result = np.argmax(output_data, axis=-1)

    print(f'Prediction: {LABELS[result]}')
    print(f'Probability: {output_data[result]:.2f}')


if __name__ == '__main__':
    main()