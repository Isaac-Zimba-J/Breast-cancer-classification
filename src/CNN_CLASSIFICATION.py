# code has 70 20 and 10 split ratio between train, validate and test data

import numpy as np
import pandas as pd
import os
import tensorflow as tf 
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt 

from sklearn.metrics import confusion_matrix
import seaborn as sns

ImageSize = (256, 256)
BatchSize = 32

strategy = tf.distribute.OneDeviceStrategy(device="/gpu:1")

dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "C:/Users/Zaac/Desktop/Workspace/python/Tensorflow/Breast cancer/Dataset",
    shuffle=True,
    image_size=ImageSize,
    batch_size=BatchSize,
)

class_names = dataset.class_names

def get_dataset(ds, trainData=0.7, ValidateData=0.2, testData=0.1, shuffle=True, shuffle_size=10000):
    ds_size = len(ds)
    if shuffle:
        ds = ds.shuffle(shuffle_size, seed=8)
    trainSize = int(trainData * ds_size)
    validateSize = int(ValidateData * ds_size)
    train_ds = ds.take(trainSize)
    validate_ds = ds.skip(trainSize).take(validateSize)
    test_ds = ds.skip(trainSize).take(validateSize)

    return train_ds, validate_ds, test_ds

with strategy.scope():
    train_ds, validate_ds, test_ds = get_dataset(dataset)

    train_ds = train_ds.cache().shuffle(10000).prefetch(buffer_size=tf.data.AUTOTUNE)
    validate_ds = validate_ds.cache().shuffle(10000).prefetch(buffer_size=tf.data.AUTOTUNE)
    test_ds = test_ds.cache().shuffle(10000).prefetch(buffer_size=tf.data.AUTOTUNE)

    resize_and_rescale = tf.keras.Sequential([
        layers.experimental.preprocessing.Resizing(256,256),
        layers.experimental.preprocessing.Rescaling(1.0/256)
    ])

    data_augmentation = tf.keras.Sequential([
        layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
        layers.experimental.preprocessing.RandomRotation(0.2),
    ])

    #model build CNN
    num_classes = 4
    input_shape = (BatchSize, 256, 256, 3)
    model3 = models.Sequential([
        resize_and_rescale,
        data_augmentation,
        layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, kernel_size=(3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(128, kernel_size=(3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(128, kernel_size=(3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax'),
    ])
    model3.build(input_shape=input_shape)

    model3.summary()

    model3.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )

    history = model3.fit(
        train_ds,
        epochs=40,
        batch_size=BatchSize,
        verbose=1,
        validation_data=validate_ds
    )

    model3.save('model/test3/model')
    model3.save('model/test3/model.h5')

    scores = model3.evaluate(test_ds)
    print(scores)

    history_data = pd.DataFrame(history.history)

    #this graph compares the training loss to that of validation loss
    plt.figure(figsize=(20,5))

    plt.subplot(1,2,1)
    plt.plot(history_data.loss, label = "Training loss")
    plt.plot(history_data.val_loss, label = "Validation loss")
    plt.xlabel("Epochs ( iterations )")
    plt.ylabel("Loss")
    plt.title("Lossess")
    plt.grid()
    plt.legend()
    plt.show()

    plt.subplot(1,2,2)
    plt.plot(history_data.accuracy, label="Training accuracy")
    plt.plot(history_data.val_accuracy, label="Validation accuracy")
    plt.xlabel("Epochs ( iterations )")
    plt.ylabel("Accuracy")
    plt.title("Accuracy")
    plt.grid()
    plt.legend()
    plt.show()




    # prediction func 
    def prediction(model, image):
        image_array = tf.keras.preprocessing.image.img_to_array(image)
        image_array = tf.expand_dims(image_array, 0)
        predictions = model.predict(image_array)
        predicted_class = class_names[np.argmax(predictions[0])]
        precision = round(100 * (np.max(predictions[0])), 2)
        return predicted_class, precision

    true_labels = []
    predicted_labels = []

    plt.figure(figsize=(15,15))
    for images, labels in test_ds.take(1):
        for i in range(12):
            
            ax = plt.subplot(3,4,i + 1)
            plt.imshow(images[i].numpy().astype('uint8'))
            
            
            predicted_class, precision = prediction(model3, images[i].numpy())
            actual_class = class_names[labels[i]]
            plt.title(f"Actual : {actual_class}, \n Predicted class : {predicted_class},\n Precision: {precision}%")
            plt.axis("off")
            
            true_labels.append(actual_class)
            predicted_labels.append(predicted_class)

    cm = confusion_matrix(true_labels, predicted_labels, labels=class_names)
    sns.heatmap(cm, annot=True, cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.show()