import tensorflow as tf
import json
from PIL import Image, ImageDraw, ImageOps

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

import numpy as np

from analyse_filtresv2 import layer_stats, plot_beta_hist
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
N_LAYERS = 1
batch_size = 32

def create_model():
    model = models.Sequential()
    model.add(layers.InputLayer(shape=(img_height, img_width, 1)))
    model.add(layers.Conv2D(8, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(2, activation='tanh'))
    return model

model2 = create_model()
model2.load_weights("./model2.keras")
model2_untrained = create_model()
model2_untrained.load_weights("./model2_untrained.keras")

def charger(split, dossier="dataset2"):
    m = json.load(open(f"{dossier}/{split}/metadata.json", encoding="utf-8"))
    X = np.stack([np.array(Image.open(f"{dossier}/{split}/{e['fichier']}"))
                  for e in m["images"]])[..., None] / 255.0          # (N, 32, 32, 1)
    y = np.array([[e["cos"], e["sin"]] for e in m["images"]], dtype="float32")
    return X.astype("float32"), y

X_train, y_train = charger("train")
X_val, y_val = charger("val")
X_test, y_test = charger("test")




def compile_and_train_model(model, X_train, y_train, X_val, y_val, epochs=EPOCHS):
    model.compile(optimizer='adam',
                  loss='mse',
                  metrics=['accuracy'])
    print(model.summary())
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs)
    printLossAcc(history, n_epoch=EPOCHS)
    save_dir = "./model2.keras"
    model.save(save_dir)
    return history

def save_untrained_model(model, filepath):
    model.save(filepath)

def evaluate_model(model, X_test, y_test):
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {test_acc:.4f}, Test loss: {test_loss:.4f}")
    y_pred = model.predict(X_test)
    angle_true = np.degrees(np.arctan2(y_test[:, 1], y_test[:, 0]))
    angle_pred = np.degrees(np.arctan2(y_pred[:, 1], y_pred[:, 0]))
    err = np.abs((angle_pred - angle_true + 180) % 360 - 180)
    print(f"Erreur angulaire moyenne : {err.mean():.2f}°  |  médiane : {err.median() if hasattr(err,'median') else np.median(err):.2f}°")
    plot_filtres(model, name="filtres_model2")
    return test_loss, test_acc, err.mean(), err.median()


def print_all_stats(model):
    print(f"Stats pour le modèle {model.name}:")
    stats = [layer_stats(model, l) for l in range(N_LAYERS)]  # couche 0 = 1ère couche de CNN
    for l in range(N_LAYERS):
        print(f"Couche {l}: β = {stats[l]['beta'].mean():.3f}")
        print(f"Couche {l}: Σ = {stats[l]['sigma'].mean():.3f}")
        print(f"Couche {l}: ||fs|| = {stats[l]['fs_signed'].mean():.3f}")
        print(f"Couche {l}: ||gx|| = {np.abs(stats[l]['gx']).mean():.3f}")
        print(f"Couche {l}: ||gy|| = {np.abs(stats[l]['gy']).mean():.3f}")
        print(f"Couche {l}: DCT = {np.abs(stats[l]['dct']).mean():.3f}")



def plot_beta_hist_comparison(model_trained, model_untrained):
    stats_trained = [layer_stats(model_trained, l) for l in range(N_LAYERS)]
    stats_untrained = [layer_stats(model_untrained, l) for l in range(N_LAYERS)]
    plot_beta_hist(stats_untrained, stats_trained, N_LAYERS, name="beta_comparison", titre="Comparaison des β : modèle entraîné vs non entraîné")

plot_filtres(model2, name="filtres_model2")
plot_filtres(model2_untrained, name="filtres_model2_untrained")

print_all_stats(model2)
print_all_stats(model2_untrained)

plot_beta_hist_comparison(model2, model2_untrained)
