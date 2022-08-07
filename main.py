import os
import numpy as np
import pandas as pd
import cv2

#gets away the text from tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

def load_model():
    old_model = tf.keras.models.load_model('saved/saved_model')

    # #Check its architecture
    # old_model.summary()
    return old_model

def load_data():
    data = pd.read_csv('train_data/answers.csv')
    print(data)
    
    # mnist = tf.keras.datasets.mnist

    # (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # return x_train / 255.0, x_test / 255.0, y_train, y_test    

def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10)
    ])
    return model

def train_model(model, x_train, y_train):
    predictions = model(x_train[:1]).numpy()
    print(predictions)

    tf.nn.softmax(predictions).numpy()
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    loss_fn(y_train[:1], predictions).numpy()

    model.compile(optimizer='adam',
                loss=loss_fn,
                metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=5)
    return model

def save_model(model):
    model.save("saved/saved_model")

def test_model(model, x_test, y_test):
    model.evaluate(x_test,  y_test, verbose=2)

def prob_model(model, x_test):
    probability_model = tf.keras.Sequential([
        model,
        tf.keras.layers.Softmax()
    ])
    probability_model(x_test[:5])

def test_model_with_pictures_from_datasets(model, x_test, y_test):
    for i in range(10):       
        img = x_test[i]
        img = cv2.resize(img, (700, 700))
        
        number = y_test[i]
        guessed_number = model.predict(x_test)

        cv2.imshow(f"Number was {number}. Guessed number was {np.argmax(guessed_number[i])}", img)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    x_train, x_test, y_train, y_test = load_data()
    model = create_model()
    train_model(model, x_train, y_train)
    test_model(model, x_test, y_test)
    save_model(model)
    prob_model(model, x_test)