import deepxde as dde 
import numpy as np
import matplotlib.pyplot as plt 

import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import tensorflow.keras as keras

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import tensorflow as tf

# Generate the training data (cuz we know the actual solution)
datasize = 100

x = np.linspace(0, 200, datasize).reshape(-1,1) #time 
y = np.sin(x) # displacement

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)


# Using normal neural network to get it's output

model = tf.keras.models.Sequential([
keras.Input(shape=(1,)),
Dense(32, kernel_initializer='normal', activation="relu"),
Dense(1, kernel_initializer='normal', activation="relu")
])

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])


model.fit(x_train, y_train, batch_size=32, epochs=10, verbose=1)
score = model.evaluate(x_test, y_test, verbose=0)


print(score)
