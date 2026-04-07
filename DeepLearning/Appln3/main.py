# tensorflow is the deep learning library
# engine
import tensorflow as tf

# easy interface to build deep learning models
# steering / driver
from tensorflow import keras

# import numpy
import numpy as np

# import fastapi
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

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

class InputData(BaseModel):
    value:float

@app.get("/")
def home():
    return {"message":"welcome to get request !!!"}


@app.post("/predict")
def predict(data:InputData):
    res = model.predict(np.array([[data.value]]))
    return {
        "prediction" : float(res[0][0])
    }

