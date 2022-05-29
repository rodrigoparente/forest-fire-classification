import json
import boto3
from io import BytesIO

import tflite_runtime.interpreter as tf
import numpy as np
from PIL import Image


LABELS = {
    0: 'With Fire',
    1: 'Without Fire' 
}

s3 = boto3.client('s3')


def lambda_handler(event, context):

    print(event)

    interpreter = tf.Interpreter(model_path='./model.tflite')
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    _, height, width, _ = input_details[0]['shape']

    bucket_name = event['Records'][0]['s3']['bucket']['name']
    key = event['Records'][0]['s3']['object']['key']

    file_byte_string = s3.get_object(Bucket=bucket_name, Key=key)['Body'].read() 
    image = Image.open(BytesIO(file_byte_string))
    image = image.resize((width, height))

    # subtracting the mean RGB value, because the vgg16 requires
    image = image - np.mean(image)

    tensor_index = input_details[0]['index']
    input_tensor = np.array(np.expand_dims(image, axis=0), dtype=np.float32)
    
    interpreter.set_tensor(tensor_index, input_tensor)

    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])

    output_data = np.squeeze(output_data)
    result = np.argmax(output_data, axis=-1)

    print(LABELS[result])
    print(output_data[result])

    # return {
    #     "Label": LABELS[result],
    #     "Probability": f'{output_data[result]:.2f}'
    # }
