import numpy as np
import pandas as pd
# machine learning library provided predefined datasets 
from sklearn.datasets import load_breast_cancer
# divid dataset into 1) training 2) testing
from sklearn.model_selection import train_test_split
# StandardScalar - used to implement the normalization
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
import matplotlib.pyplot as plt

data = load_breast_cancer()
X = data.data
y = data.target

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = keras.Sequential([
    keras.layers.Dense(32,activation='relu'),
    keras.layers.Dense(16,activation='relu'),
    keras.layers.Dense(1,activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history = model.fit(X_train,y_train,epochs=100,validation_data=(X_test,y_test))

loss,accuracy = model.evaluate(X_test,y_test)
print(accuracy)

if model.predict(X_test[0].reshape(1,-1))[0][0] > 0.5:
    print("Safe")
else:
    print("Danger")

plt.plot(history.history["accuracy"])
print(history.history["accuracy"])
plt.plot(history.history['val_accuracy'])
print("-------------------")
print(history.history['val_accuracy'])
plt.title('Accuracy Graph')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train','Test'])
plt.show()