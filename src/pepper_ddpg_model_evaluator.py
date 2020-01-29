# -*- coding: utf-8 -*-
# Run with newer tensorflow version.
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
model.add(Dense(3, input_dim=3, kernel_initializer='uniform', activation='elu'))
model.add(Dense(2, kernel_initializer='uniform', activation='elu'))
model.add(Dense(2, kernel_initializer='uniform', activation='elu'))

# Compile model
model.compile(loss='mean_absolute_error', optimizer='SGD', metrics=['accuracy'])

# Fit the model
# Epoch Anzahl, Batch Size
print("Training Model")
model.fit(X, Y, epochs=150, batch_size=256, verbose=1, shuffle=1, validation_split=0.7)

print("Evaluate")
# evaluate the model
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

print("Predict")
pred = numpy.array([[-0.0153398513794, 237, 0.4]], numpy.float)
print(model.predict(pred, batch_size=None, verbose=0, steps=None, callbacks=None, max_queue_size=10, workers=1,
                    use_multiprocessing=False))
print("Should be: 224,-10124")
