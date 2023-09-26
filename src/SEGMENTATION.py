"""
Created on Sun Aug  6 13:33:59 2023

@author: Zaac
"""

import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from keras import layers, models
import pathlib
import os
import cv2
import string
from keras.metrics import Accuracy, Precision, Recall
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.metrics import f1_score
import seaborn as sns
import pandas as pd
import pickle 


#preparation of the dataset (images)
class readDataset:
    def __init__(self, datasetpath, imageShape):
        self.datasetpath = datasetpath
        self.imageShape = imageShape
    def imagesPath(self, folder, name):
        images = list(pathlib.Path(os.path.join(self.datasetpath, 
                                                folder)).glob('*{}.*'.format(name)))
        return images
    def readImages(self, images, channels):
        listImages = []
        images = np.sort(images)
        for image in images:
            image = tf.io.read_file(str(image))
            image = tf.image.decode_png(image, channels = channels)
            image = tf.image.resize(image, self.imageShape)
            image/= 255
            listImages.append(image)
        return listImages
    def allDataset(self, label):
        images = self.readImages(self.imagesPath(label, name = ')'), channels = 3)
        masks = np.array(self.readImages(self.imagesPath(label, name = 'mask'), channels = 1))
        masks = (masks >= 0.5).astype('int32')
        return np.array(images), masks
    def finalDataset(self, labels):
        images_benign, masks_benign = self.allDataset(labels[0])
        images_malignant, masks_malignant = self.allDataset(labels[1])
        images_normal, masks_normal = self.allDataset(labels[2])
        images = np.vstack([images_benign, images_malignant, images_normal])
        masks = np.vstack([masks_benign, masks_malignant, masks_normal])
        labels = np.hstack([np.ones(shape = (len(images_benign),))*0,
                           np.ones(shape = (len(images_malignant), ))*1, 
                           np.ones(shape = (len(images_normal), ))*2])
        return images, masks, labels
    def dataAugmentation(self, images, masks, labels):
        imagesupdate = []
        masksupdate = []
        labelsupdate = []
        for image, mask, label in zip(images, labels, masks):
            image1 = tf.image.adjust_contrast(image, contrast_factor = 2)
            image2 = tf.image.adjust_brightness(image, delta = 0.3)
            imagesupdate.append(image), masksupdate.append(mask), labelsupdate.append(label)
            imagesupdate.append(image1), masksupdate.append(mask), labelsupdate.append(label)
            imagesupdate.append(image2), masksupdate.append(mask), labelsupdate.append(label)
        return np.array(imagesupdate), np.array(masksupdate), np.array(labelsupdate)

#model
datasetpath = 'Dataset'
datasetObject = readDataset(datasetpath, [128, 128])



images, masks, labels = datasetObject.finalDataset(['benign', 'malignant', 'normal'])

#splitting the dataset
images, masks, labels = datasetObject.dataAugmentation(images, labels, masks)
images.shape, masks.shape, labels.shape


# #show some images 
# def showImagesWithMask(images, masks, labels):
#     plt.figure(figsize = (12, 12))
#     for i in range(len(images)):
#         plt.subplot(8, 8, (i + 1))
#         plt.imshow(images[i])
#         plt.imshow(masks[i], alpha = 0.3, cmap = 'jet')
#         plt.title(labels[i])
#     plt.legend()

# showImagesWithMask(images[:64], masks[:64], labels[:64])

def showImagesWithMask(images, masks, labels):
    plt.figure(figsize=(12, 12))
    for i in range(len(images)):
        plt.subplot(8, 8, (i + 1))
        plt.imshow((images[i] * 255).astype(np.uint8))  # Rescale back to [0, 255]
        plt.imshow((masks[i] * 255).astype(np.uint8), alpha=0.3, cmap='jet')  # Rescale masks
        plt.title(labels[i])
    plt.show()  # Show the plot
# showImagesWithMask(images[:64], masks[:64], labels[:64])

#divide the dataset into train and test
randomIndexs = np.random.randint(0, len(images), size = (len(images), ))
images = images[randomIndexs]
masks = masks[randomIndexs]
labels = labels[randomIndexs]
images.shape, masks.shape, labels.shape

valid = 500
trainDataset = [images[:images.shape[0] - valid], 
         masks[:images.shape[0] - valid], 
         labels[:images.shape[0] - valid]]
validDataset = [images[images.shape[0] - valid:], 
         masks[images.shape[0] - valid:], 
         labels[images.shape[0] - valid:]]

trainDataset[0].shape, trainDataset[1].shape, trainDataset[2].shape
validDataset[0].shape, validDataset[1].shape, validDataset[2].shape

#neural network architecture 
def convolution(inputs, padding, strides, filter, kernel_size):
    x = inputs
    y = layers.Conv2D(filter, kernel_size = 1, padding = padding, 
                     strides = strides, 
                     kernel_regularizer = tf.keras.regularizers.L2(0.001))(x)
    x = layers.Conv2D(filter, kernel_size = kernel_size, padding = padding, 
                     strides = strides, 
                     kernel_regularizer = tf.keras.regularizers.L2(0.001))(y)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filter, kernel_size = kernel_size, padding = padding, 
                     strides = strides, 
                     kernel_regularizer = tf.keras.regularizers.L2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.add([x, y])
    x = layers.Activation('relu')(x)
    return x


def encoder(inputs, filter):
    correlation = convolution(inputs, padding = 'same', strides = 1, filter = filter, 
                             kernel_size = 5)
    downsample = layers.AveragePooling2D()(correlation)
    return correlation, downsample


def decoder(inputs, skip_connection, filter):
    upsample = layers.Conv2DTranspose(filter, 5, padding = 'same', strides = 2, 
                                     kernel_regularizer = tf.keras.regularizers.L2(0.001))(inputs)
    upsample = layers.Activation('relu')(upsample)
    upsample = layers.BatchNormalization()(upsample)
    connection = layers.average([upsample, skip_connection])
    correlation = convolution(connection, padding = 'same', strides = 1, filter = filter, 
                             kernel_size = 5)
    return correlation

input = layers.Input(shape = (128, 128, 3))
filter = 32
corr1, downsample1 = encoder(input, filter)
corr2, downsample2 = encoder(downsample1, filter*2)
corr3, downsample3 = encoder(downsample2, filter*4)
corr4, downsample4 = encoder(downsample3, filter*8)
downsample4 = convolution(downsample4, padding = 'same', strides = 1, filter = filter*8, 
                         kernel_size = 5)
features_vector_1 = layers.GlobalAveragePooling2D()(downsample4)
features_vector_2 = layers.Flatten()(downsample4)
features_vector_2 = layers.Dropout(0.5)(features_vector_2)
features_vector_1 = layers.Dropout(0.5)(features_vector_1)
encoder_x = layers.Dense(64, name = 'latent_space', 
                kernel_regularizer = tf.keras.regularizers.L2(0.001))(features_vector_1)
x = layers.Dense(downsample4.shape[1]*downsample4.shape[2]*downsample4.shape[3], 
                kernel_regularizer = tf.keras.regularizers.L2(0.001))(encoder_x)
x = layers.Reshape((downsample4.shape[1], downsample4.shape[2], downsample4.shape[3]), 
                  name = 'reshape')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.4)(x)
decoder_corr1 = decoder(x, corr4, filter*8)
decoder_corr2 = decoder(decoder_corr1, corr3, filter*4)
decoder_corr3 = decoder(decoder_corr2, corr2, filter*2)
decoder_corr4 = decoder(decoder_corr3, corr1, filter)
output = layers.Conv2DTranspose(1, 5, padding = 'same', strides = 1)(decoder_corr4)
output = layers.Activation('sigmoid', name = 'UNET')(output)
labelOutput = layers.Dense(32, activation = 'relu')(features_vector_2)
labelOutput = layers.BatchNormalization()(labelOutput)
labelOutput = layers.Dropout(0.5)(labelOutput)
labelOutput = layers.Dense(16, activation = 'relu')(labelOutput)
labelOutput = layers.BatchNormalization()(labelOutput)
labelOutput = layers.Dropout(0.5)(labelOutput)
labelOutput = layers.Dense(1, name = 'label')(labelOutput)
m = models.Model(inputs = input, outputs = [output, labelOutput])
m.compile(loss = [tf.keras.losses.BinaryFocalCrossentropy(), 'mae'], 
          optimizer = tf.keras.optimizers.Adam(learning_rate = 0.00001), 
          metrics = ['accuracy', Precision(name = 'precision'), Recall(name = 'recall')], 
          )
m.summary()

tf.keras.utils.plot_model(m,
                            show_shapes=True,
                            show_layer_names=True,)

history = m.fit(trainDataset[0], [trainDataset[1], trainDataset[2]], epochs = 260, 
               validation_data = (validDataset[0], [validDataset[1], validDataset[2]]), 
               batch_size = 8, callbacks = [
                   tf.keras.callbacks.EarlyStopping(patience = 12, monitor = 'val_loss', 
                                                   mode = 'min', restore_best_weights = True)
               ])

m.evaluate(trainDataset[0], [trainDataset[1], trainDataset[2]])

m.evaluate(validDataset[0], [validDataset[1], validDataset[2]])

metrics = ['loss', 'UNET_accuracy', 'UNET_recall', 'UNET_precision']
plt.figure(figsize = (12, 16))
for i in range(4):
    plt.subplot(4, 1, (i + 1))
    plt.plot(history.history['{}'.format(metrics[i])], label = '{}'.format(metrics[i]))
    plt.plot(history.history['val_{}'.format(metrics[i])], label = 'val_{}'.format(metrics[i]))
    plt.title('{}'.format(metrics[i]))
plt.legend()

masks_pred, _ = m.predict(validDataset[0])

masks_pred = (masks_pred >= 0.5).astype('int32')
np.unique(masks_pred), np.unique(validDataset[1])

masks_pred = layers.Flatten()(masks_pred)
masks_pred  = np.reshape(masks_pred, -1)
masks_actual = layers.Flatten()(validDataset[1])
masks_actual  = np.reshape(masks_actual, -1)
masks_pred.shape, masks_actual.shape

print(classification_report(masks_actual, masks_pred))

tn, fp, fn, tp = confusion_matrix(masks_actual, masks_pred).ravel()

metrics = pd.DataFrame([accuracy_score(masks_actual, masks_pred)*100,
                        precision_score(masks_actual, masks_pred)*100,
                        recall_score(masks_actual, masks_pred)*100,
                        f1_score(masks_actual, masks_pred)*100,
                        (tn / (tn+fp))*100,
                        (2*tp/(2*tp + fp + fn))*100,
                        (tp/(tp + fn))*100],
                        index = ['Accuracy Score', 'Precision Score',
                                'Recall Score', 'F1 Score', 'specificity', 
                                'dice Score', 'sensitivity'], 
                        columns = ['Metrics For Validation Data'])
metrics.head(n= 8)

def segmentation(data):
    masks_pred, _ = m.predict(data)
    masks_pred = np.array(masks_pred)
    masks_pred = (masks_pred >= 0.5).astype('int32')
    return masks_pred

valid_masks = segmentation(validDataset[0])
valid_masks.shape

def draw(images, masks, y_pred):
  plt.figure(figsize = (12, 25))
  index = -1
  n = np.random.randint(y_pred.shape[0])
  for i in range(60):
    plt.subplot(10, 6, (i + 1))
    if index == -1:
      plt.imshow(images[n])
      plt.title('Image')
      index = 0
    elif index == 0:
      plt.imshow(images[n])
      plt.imshow(masks[n], alpha = 0.3, cmap = 'jet')
      plt.title('Original Mask')
      index = 1
    elif index == 1:
      plt.imshow(images[n])
      plt.imshow(np.reshape(y_pred[n], (128, 128)), alpha = 0.3, cmap = 'jet')
      plt.title('Predict Mask')
      index = -1
      n = np.random.randint(y_pred.shape[0])
  plt.legend()
  
  draw(validDataset[0], validDataset[1], valid_masks)
  
  draw(validDataset[0], validDataset[1], valid_masks)
  
  draw(validDataset[0], validDataset[1], valid_masks)
  
train_masks = segmentation(trainDataset[0])
train_masks.shape

draw(trainDataset[0], trainDataset[1], train_masks)

draw(trainDataset[0], trainDataset[1], train_masks)

m.save('work/breast_cancer_segmentation.h5')