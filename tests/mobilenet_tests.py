# hiding tensorflow warnings
import os
import warnings
from timeit import default_timer as timer

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

warnings.filterwarnings("ignore")

# third-party Imports
import numpy as np

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

import tensorflow as tf
import tflite_runtime.interpreter as tflite

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def fmt_list(value):
    return ' '.join(str(i) for i in value)


def fmt_matrix(cm):
    tmp = list()
    for row in cm.tolist():
        for col in row:
            tmp.append(col)
    return tmp


def to_file(filename, text):
    # check and create folder it doesn't exists
    dirs = os.path.split(filename)[0]
    if not os.path.exists(dirs):
        os.makedirs(dirs)

    with open(filename, 'a') as f:
        f.write(f'{text}\n')


def load_datasets():
    home_dir = os.path.expanduser('~')
    project_dir = f'{home_dir}/Projects/fire'

    train_path = f'{project_dir}/data/fire-vs-without-fire/train'
    valid_path = f'{project_dir}/data/fire-vs-without-fire/valid'
    test_path = f'{project_dir}/data/fire-vs-without-fire/test'

    train_batches = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input) \
        .flow_from_directory(directory=train_path, target_size=(224, 224),
                             classes=['with_fire', 'without_fire'], batch_size=10)

    valid_batches = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input) \
        .flow_from_directory(directory=valid_path, target_size=(224, 224),
                             classes=['with_fire', 'without_fire'], batch_size=10)

    test_batches = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input) \
        .flow_from_directory(directory=test_path, target_size=(224, 224),
                             classes=['with_fire', 'without_fire'], batch_size=10)

    return train_batches, valid_batches, test_batches


def unpack_batch(test_batches):
    n_samples = test_batches.n
    batch_size = test_batches.batch_size

    n_inter = n_samples / batch_size
    n_inter = int(n_inter)

    imgs_list = list()
    y_true = list()

    for _ in range(n_inter):
        imgs, labels = next(test_batches)

        for img in imgs:
            imgs_list.append(np.expand_dims(img, axis=0))

        y_true += list(np.argmax(labels, axis=-1))

    return imgs_list, y_true


def exec_full_model(train_batches, valid_batches, test_imgs, test_labels, n_epochs=1):
    # import mobile net

    # include_top=False will return the model without the dense layers,
    # in order to let you make your own dense layers and make your
    # own classification to suit your needs.

    mobilenet_base =\
        tf.keras.applications.mobilenet_v2.MobileNetV2(
            input_shape=(224, 224, 3), include_top=False, weights='imagenet')

    model = Sequential()
    model.add(mobilenet_base)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(units=2, activation='softmax'))

    # fine-tuned MobileNet model

    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    model.fit(x=train_batches,
              steps_per_epoch=len(train_batches),
              validation_data=valid_batches,
              validation_steps=len(valid_batches),
              epochs=n_epochs,
              verbose=0)

    # predicting

    t_predict = timer()

    y_pred = list()
    for img in test_imgs:
        predictions = model.predict(img, verbose=0)
        y_pred += list(np.argmax(predictions, axis=-1))

    t_predict = timer() - t_predict

    # calculating metrics

    metrics = {
        'acc': 0,
        'time': 0,
        'precision': 0,
        'recall': 0,
        'f1': 0,
        'cm': 0
    }

    metrics['acc'] = accuracy_score(test_labels, y_pred)
    metrics['time'] = t_predict

    metrics['precision'] = precision_score(test_labels, y_pred)
    metrics['recall'] = recall_score(test_labels, y_pred)
    metrics['f1'] = f1_score(test_labels, y_pred)

    cm = confusion_matrix(test_labels, y_pred, labels=[0, 1], normalize='true')
    metrics['cm'] = fmt_list(fmt_matrix(cm))

    return model, metrics


def exec_tflite_model(full_model, test_imgs, test_labels):
    # convert model to tflite

    converter = tf.lite.TFLiteConverter.from_keras_model(full_model)
    tflite_model = converter.convert()

    # inference using tflite

    interpreter = tflite.Interpreter(model_content=tflite_model)

    input_details = interpreter.get_input_details()
    input_index = input_details[0]['index']

    output_details = interpreter.get_output_details()
    output_index = output_details[0]['index']

    interpreter.allocate_tensors()

    # predict

    t_predict = timer()

    y_pred = list()
    for img in test_imgs:

        interpreter.set_tensor(input_index, img)

        interpreter.invoke()

        output_data = interpreter.get_tensor(output_index)

        output_data = np.squeeze(output_data)

        y_pred.append(np.argmax(output_data, axis=-1))

    t_predict = timer() - t_predict

    # calculating metrics

    metrics = {
        'acc': 0,
        'time': 0,
        'precision': 0,
        'recall': 0,
        'f1': 0,
        'cm': 0
    }

    metrics['acc'] = accuracy_score(test_labels, y_pred)
    metrics['time'] = t_predict

    metrics['precision'] = precision_score(test_labels, y_pred)
    metrics['recall'] = recall_score(test_labels, y_pred)
    metrics['f1'] = f1_score(test_labels, y_pred)

    cm = confusion_matrix(test_labels, y_pred, labels=[0, 1], normalize='true')
    metrics['cm'] = fmt_list(fmt_matrix(cm))

    return metrics


def main():

    for _ in range(30):

        # loading images from directory
        train_batches, valid_batches, test_batches = load_datasets()

        # transforming the test_batches generator into an array
        test_batches, y_true = unpack_batch(test_batches)

        # executing full model
        model, metrics = exec_full_model(
            train_batches, valid_batches, test_batches, y_true)

        for key, value in metrics.items():
            to_file(f'results/{key}s.txt', value)

        # executing tflite model
        metrics = exec_tflite_model(model, test_batches, y_true)

        for key, value in metrics.items():
            to_file(f'results/tflite-{key}s.txt', value)


if __name__ == '__main__':
    main()
