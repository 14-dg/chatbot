import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('all')

import os
#gets away the text from tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Activation, Dropout
#from tensorflow.python.keras.optimizers import SGD

lemmatizer = WordNetLemmatizer()

intents = json.loads(open('intents.json').read())

words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append(((word_list), intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

print(documents)
words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))

print(words)