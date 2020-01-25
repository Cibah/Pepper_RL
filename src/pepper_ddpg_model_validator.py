# -*- coding: utf-8 -*-
from keras import Model
from keras.layers import Dense
import numpy
from Settings import *
import json

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load pima indians dataset
file = open(TRAINING_FILE)
datasets = json.load(file)['steps']
file.close()

INPUT1 = []
INPUT2 = []
INPUT3 = []
OUTPUT1 = []
OUTPUT2 = []

for data in datasets:
    INPUT1.append(data['az'])
    INPUT2.append(data['ad'])
    INPUT3.append(data['actionR'])
    OUTPUT1.append(data['fd'])
    OUTPUT2.append(data['rw'])

in1 = numpy.ndarray(INPUT1)
in2 = numpy.ndarray(INPUT2)
in3 = numpy.ndarray(INPUT3)
out1 = numpy.ndarray(OUTPUT1)
out2 = numpy.ndarray(OUTPUT2)

# split into input (X) and output (Y) variables


# create model
# model = Sequential()
model = Model(inputs=[in1, in2, in3], outputs=[out1, out2])
# Layer 1: 12 Neuronen, 4 Eingänge, Gewichtung uniform=random, aktivierungsfunktion relu
model.add(Dense(3, input_dim=3, kernel_initializer='uniform', activation='relu', input_shape=(len(INPUT1),)))
# Layer 2: 8 Neuronen, Gewichtung uniform=random, aktivierungsfunktion relu
model.add(Dense(2, kernel_initializer='uniform', activation='relu'))
# Layer 3: 1 Neuronen, Gewichtung uniform=random, aktivierungsfunktion sigmoid
model.add(Dense(2, kernel_initializer='uniform', activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='Adadelta', metrics=['accuracy'])

# Fit the model
# Epoch Anzahl, Batch Size
# model.fit(X, Y, epochs=250, batch_size=100, verbose=0)
model.fit(inputs=[in1, in2, in3], outputs=[out1, out2], epochs=250, batch_size=100, verbose=0)

# evaluate the model
# scores = model.evaluate(X, Y)
# scores = model.evaluate(INPUT1, INPUT2, INPUT3, OUTPUT1, OUTPUT2)
# print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
