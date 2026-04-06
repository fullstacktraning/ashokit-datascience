# tensorflow is the deep learning library
# engine
import tensorflow as tf

# easy interface to build deep learning models
# steering / driver
from tensorflow import keras

# import numpy
import numpy as np

# create the model
model = keras.Sequential([
    keras.layers.Dense(3,activation='relu'),
    keras.layers.Dense(1)
])

# compile
model.compile(
    optimizer='adam',
    loss='mean_squared_error'
)

X = np.array([[1],[2],[3],[4]],dtype=float)
y = np.array([[2],[4],[6],[8]],dtype=float)

model.fit(X,y,epochs=500)

# epochs - 1 (Wrong)
# epochs - 100 (Better)
# epochs - 500 (Good)

# print( model.predict([5.0]) )
print(model.predict(np.array([[5.0]])))

