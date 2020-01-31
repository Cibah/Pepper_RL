# -*- coding: utf-8 -*-
# Run with newer tensorflow version.
import json

from keras import Sequential
from keras.layers import Dense
import numpy
from Settings import *
from ddpg.ddpg import getReward

seed = 7
numpy.random.seed(seed)

dataset = numpy.loadtxt("training_set.csv", delimiter=",")
X = dataset[:, 0:3]
Y = dataset[:, 4:5]

batches = [8, 16, 32, 64, 128, 256, 512, 1024]

# X = normalize_fixed(X, [[-0.25, 0.4], [0, 500], [-0.25, 0.4]], [[0,1]])

# for epoch in range(10, 2500, 50):


# create model
model = Sequential()
model.add(Dense(3, input_dim=3, kernel_initializer='uniform', activation='elu'))
model.add(Dense(2, kernel_initializer='uniform', activation='elu'))
model.add(Dense(1, kernel_initializer='uniform', activation='elu'))

# Compile model
model.compile(loss='mean_absolute_error', optimizer='SGD', metrics=['accuracy'])

# Fit the model
# Epoch Anzahl, Batch Size
print("Training Model")
model.fit(X, Y, epochs=150, batch_size=256, verbose=0, shuffle=1, validation_split=0.7)

fileJSON = open(TRAINING_FILE)
datasets = json.load(fileJSON)['steps']
fileJSON.close()
success = 0

for set in datasets:
    pred = numpy.array([[set['az'], set['ad'], set['actionR']]], numpy.float)
    result = (model.predict(pred, batch_size=None, verbose=0, steps=None, callbacks=None, max_queue_size=10, workers=1,
                            use_multiprocessing=False))
    predictedDelta = result[0][0]
    realDelta = set["fd"]
    accuracy = 0.1  # +-10%

    if predictedDelta < realDelta * (1 + accuracy) and predictedDelta > realDelta * (1 - accuracy):
        success += 1
        # print(str(realDelta * (1 - accuracy)) + " < " + str(predictedDelta) + " > " + str(realDelta * (1 + accuracy)))

print("Accuracy: " + str(success * 100 / len(datasets)))
