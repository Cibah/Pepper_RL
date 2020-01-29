# -*- coding: utf-8 -*-
from keras import Sequential
from keras.layers import Dense
import numpy

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load pima indians dataset

dataset = numpy.loadtxt("training_set.csv", delimiter=",")
X = dataset[:, 0:3]
Y = dataset[:, 4:6]

# create model
model = Sequential()
model.add(Dense(3, input_dim=3, kernel_initializer='uniform', activation='relu'))
model.add(Dense(2, kernel_initializer='uniform', activation='relu'))
model.add(Dense(2, kernel_initializer='uniform', activation='relu'))

# Compile model
model.compile(loss='mean_absolute_error', optimizer='SGD', metrics=['accuracy'])

# Fit the model
# Epoch Anzahl, Batch Size
print("Training Model")
model.fit(X, Y, epochs=150, batch_size=64, verbose=1, shuffle=1, validation_split=0.8)

print("Evaluate")
# evaluate the model
scores = model.evaluate(X, Y)
# scores = model.evaluate(INPUT1, INPUT2, INPUT3, OUTPUT1, OUTPUT2)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
