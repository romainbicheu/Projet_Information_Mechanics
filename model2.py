import tensorflow as tf
import json
from PIL import Image, ImageDraw, ImageOps

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

import numpy as np

from result_visualisation import printLossAcc, plot_confusion_matrix
from weight_visualisation import getWeights, plot_filtres

img_height=32
img_width = 32
dataset_directory = "dataset"
train_directory = "dataset2/train/images"
val_directory = "dataset2/val/images"
test_directory = "dataset2/test/images"
train_metadata = "dataset2/train/metadata.json"
val_metadata = "dataset2/val/metadata.json"
test_metadata = "dataset2/test/metadata.json"

EPOCHS = 50
batch_size = 32

def charger(split, dossier="dataset2"):
    m = json.load(open(f"{dossier}/{split}/metadata.json", encoding="utf-8"))
    X = np.stack([np.array(Image.open(f"{dossier}/{split}/{e['fichier']}"))
                  for e in m["images"]])[..., None] / 255.0          # (N, 32, 32, 1)
    y = np.array([[e["cos"], e["sin"]] for e in m["images"]], dtype="float32")
    return X.astype("float32"), y

X_train, y_train = charger("train")
X_val, y_val = charger("val")
X_test, y_test = charger("test")


model2 = models.Sequential()
model2.add(layers.InputLayer(input_shape=(img_height, img_width, 1)))
model2.add(layers.Conv2D(8, (3, 3), activation='relu'))
model2.add(layers.MaxPooling2D((2, 2)))
model2.add(layers.Flatten())
model2.add(layers.Dense(2, activation='tanh'))



model2.compile(optimizer='adam',
              loss='mse',
              metrics=['accuracy'])
print(model2.summary())

history = model2.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=EPOCHS)

printLossAcc(history, n_epoch=EPOCHS)


model2.evaluate(X_test, y_test)


y_pred = model2.predict(X_test)
angle_true = np.degrees(np.arctan2(y_test[:, 1], y_test[:, 0]))
angle_pred = np.degrees(np.arctan2(y_pred[:, 1], y_pred[:, 0]))
err = np.abs((angle_pred - angle_true + 180) % 360 - 180)
print(f"Erreur angulaire moyenne : {err.mean():.2f}°  |  médiane : {err.median() if hasattr(err,'median') else np.median(err):.2f}°")


plot_filtres(model2, name="filtres_model2")
