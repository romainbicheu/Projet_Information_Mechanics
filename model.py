import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

import numpy as np

from result_visualisation import printLossAcc, plot_confusion_matrix
from weight_visualisation import getWeights

img_height=32
img_width = 32
dataset_directory = "dataset"
train_directory = "dataset/train"
val_directory = "dataset/val"
test_directory = "dataset/test"
EPOCHS = 3
batch_size = 32

#MODEL PARAMETERS


train_ds = tf.keras.utils.image_dataset_from_directory(
    train_directory,
    image_size=(img_height, img_width),
    color_mode="grayscale",     
    batch_size=batch_size,
    label_mode="categorical",
    shuffle=True,
    seed=123,
    )
val_ds = tf.keras.utils.image_dataset_from_directory(
    val_directory,
    image_size=(img_height, img_width),
    color_mode="grayscale",     
    batch_size=batch_size,
    label_mode="categorical",
    shuffle=False,
    seed=123,
    )
test_ds = tf.keras.utils.image_dataset_from_directory(
    test_directory,
    image_size=(img_height, img_width),
    color_mode="grayscale",     
    batch_size=batch_size,
    label_mode="categorical",
    shuffle=False,
    seed=123,
    )
class_names = train_ds.class_names
print(class_names)
print(train_ds.element_spec)


model1 = models.Sequential()
Input1 = model1.add(layers.InputLayer(input_shape=(img_height, img_width, 1)))
Recaling1 = model1.add(layers.Rescaling(1./255))
Layer1 = model1.add(layers.Conv2D(3, (3, 3), activation='relu'))
MaxPool1 = model1.add(layers.MaxPooling2D((2, 2)))
Flatten1 = model1.add(layers.Flatten())
Output = model1.add(layers.Dense(3, activation='softmax'))



model1.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])
print(model1.summary())

history = model1.fit(train_ds, validation_data=(val_ds), epochs=EPOCHS)

printLossAcc(history, n_epoch=EPOCHS)


model1.evaluate(test_ds)


y_true = np.concatenate([y.numpy() for _, y in test_ds]).argmax(axis=1)
y_pred = model1.predict(test_ds).argmax(axis=1)

cm = confusion_matrix(y_true, y_pred)
print(cm)

plot_confusion_matrix(cm, classes=class_names, title="Confusion Matrix for Model 1")

getWeights(model1)

