# ================================
# STEP 1: Import Libraries
# ================================
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
import matplotlib.pyplot as plt

# ================================
# STEP 2: Load Dataset (NO DOWNLOAD)
# ================================
data = load_breast_cancer()

X = data.data
y = data.target

print("Dataset Loaded!")
print("Shape:", X.shape)

# ================================
# STEP 3: Train-Test Split
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ================================
# STEP 4: Normalize Data
# ================================
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ================================
# STEP 5: Build Deep Learning Model
# ================================
model = keras.Sequential([
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

# ================================
# STEP 6: Compile Model
# ================================
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ================================
# STEP 7: Train Model
# ================================
history = model.fit(
    X_train, y_train,
    epochs=100,
    validation_data=(X_test, y_test)
)

# ================================
# STEP 8: Evaluate Model
# ================================
loss, accuracy = model.evaluate(X_test, y_test)
print("\nFinal Accuracy:", accuracy)

# ================================
# STEP 9: Prediction
# ================================
sample = X_test[0].reshape(1, -1)

prediction = model.predict(sample)

print("\nPrediction Value:", prediction)

if prediction > 0.5:
    print("Benign (Safe)")
else:
    print("Malignant (Dangerous)")

# ================================
# STEP 10: Plot Graph
# ================================
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.title("Accuracy Graph")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.legend(["Train", "Test"])
plt.show()