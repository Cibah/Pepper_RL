# -*- coding: utf-8 -*-
from keras import Model, Sequential
from keras.layers import Dense
import numpy

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load pima indians dataset

dataset = numpy.loadtxt("training.csv", delimiter=",")
X = dataset[:, 0:3]
Y = dataset[:, 4]

print(X)
print(Y)

# create model
model = Sequential()
# Layer 1: 12 Neuronen, 4 Eingänge, Gewichtung uniform=random, aktivierungsfunktion relu
model.add(Dense(3, input_dim=3, kernel_initializer='uniform', activation='relu'))
# Layer 2: 8 Neuronen, Gewichtung uniform=random, aktivierungsfunktion relu
model.add(Dense(2, kernel_initializer='uniform', activation='relu'))
# Layer 3: 1 Neuronen, Gewichtung uniform=random, aktivierungsfunktion sigmoid
model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='Adadelta', metrics=['accuracy'])

# Fit the model
# Epoch Anzahl, Batch Size
model.fit(X, Y, epochs=2500, batch_size=64, verbose=0)

# evaluate the model
scores = model.evaluate(X, Y)
# scores = model.evaluate(INPUT1, INPUT2, INPUT3, OUTPUT1, OUTPUT2)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
