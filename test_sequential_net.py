from pyClarion.base.stubs import *
import keras
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import text_to_word_sequence
import random
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from typing import List, Iterable, Dict, Hashable, Mapping

d = dict()
d["red"] = 1
d["blue"] = 0
d["green"] = 0.5

d1 = dict()
d1["red"] = 0.9
d1["blue"] = 0.2
d1["green"] = 0.4


d2 = dict()
d2["red"] = 0.6
d2["blue"] = 0.3
d2["green"] = 0.2


ipt = ["red", "blue", "green"]

opt = [1, 0, 0.5]
model = Sequential()
model.add(Dense(64, input_dim=1, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


K = KerasSequentialNet(model=model, inputs=ipt,  outputs=ipt, default_strength=0.7)
print(K(d))
K.update(training_input=K.vectorizer.input2vector(d1), training_output=np.asarray(opt))
print(K(d))
K.update(training_input=K.vectorizer.input2vector(d2), training_output=np.asarray(opt))
print(K(d))





