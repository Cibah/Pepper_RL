# -*- coding: utf-8 -*-
# Run with newer tensorflow version.
import json

import keras
from keras import Sequential
from keras.layers import Dense, BatchNormalization
import numpy
from Settings import *

seed = 7
numpy.random.seed(seed)

# Get Trainingsdata
dataset = numpy.loadtxt("training_set.csv", delimiter=",")
X = dataset[:, 0:3]  # Input Data: az, ad, actionR
Y = dataset[:, 4:5]  # Output Data: fd
fileJSON = open(TRAINING_FILE)
datasets = json.load(fileJSON)['steps']
fileJSON.close()

# X = normalize_fixed(X, [[-0.25, 0.4], [0, 500], [-0.25, 0.4]], [[0,1]])

# Create the model
model = Sequential()
model.add(Dense(3, input_dim=3, kernel_initializer='uniform', activation='elu'))
model.add(BatchNormalization())
model.add(Dense(2, kernel_initializer='uniform', activation='elu'))
model.add(Dense(1, kernel_initializer='uniform', activation='elu'))

epoch = 30
batch = 32
success = 0
accuracyOfPredictions = 0.15  # +-15%

# Compile model
optimizer = keras.optimizers.SGD(learning_rate=0.01, momentum=0.0, nesterov=False)
model.compile(loss='mean_absolute_error', optimizer=optimizer, metrics=['accuracy'])

# Train model with 70% training and 30% validation split
model.fit(X, Y, epochs=epoch, batch_size=batch, verbose=0, shuffle=1, validation_split=0.3)

for set in datasets:
    pred = numpy.array([[set['az'], set['ad'], set['actionR']]], numpy.float)
    result = (model.predict(pred, batch_size=None, verbose=0, steps=None, callbacks=None, max_queue_size=10,
                            workers=1, use_multiprocessing=False))
    predictedDelta = result[0][0]
    realDelta = set["fd"]
    if realDelta * (1 + accuracyOfPredictions) > predictedDelta > realDelta * (1 - accuracyOfPredictions):
        success += 1

print("Accuracy: " + str(success * 100 / len(datasets)))
