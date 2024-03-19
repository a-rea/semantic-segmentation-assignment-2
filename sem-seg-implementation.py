import warnings
import os
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics

import tensorflow.keras as kb
from tensorflow.keras import backend
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, MaxPooling2D, Conv2D, Conv2DTranspose, BatchNormalization, Dropout, Concatenate
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.optimizers import Adam


X_train_path = "face-synthetics-glasses/train/images"
y_train_path = "face-synthetics-glasses/train/masks"

X_test_path = "face-synthetics-glasses/test/images"
y_test_path = "face-synthetics-glasses/test/masks"



def load_data(image_path, mask_path):
    images = []
    masks = []
    
    # Get list of image and mask filenames
    image_filenames = os.listdir(image_path)
    mask_filenames = os.listdir(mask_path)
    
    # Sort filenames to ensure images and masks match
    image_filenames.sort()
    mask_filenames.sort()
    
    for img_file, mask_file in zip(image_filenames, mask_filenames):
        # Load image and mask
        img = load_img(os.path.join(image_path, img_file), target_size=(256, 256))
        mask = load_img(os.path.join(mask_path, mask_file), target_size=(256, 256), color_mode='grayscale')
        
        # Convert image and mask to arrays
        img_array = img_to_array(img) / 255.0  # Normalize to [0, 1]
        mask_array = img_to_array(mask) / 255.0  # Normalize to [0, 1]
        
        images.append(img_array)
        masks.append(mask_array)
    
    # Convert lists to arrays
    images = np.array(images)
    masks = np.array(masks)
    
    return images, masks

def data_processing():
    # Load training and testing data
    X_train, y_train = load_data(X_train_path, y_train_path)
    X_test, y_test = load_data(X_test_path, y_test_path)

    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    
    
    return X_train, y_train, X_test, y_test

def eda(X_train, y_train, X_test, y_test):

    train_images = X_train.shape[0]
    test_images = X_test.shape[0]
    test_masks = y_test.shape[0]
    train_masks = y_train.shape[0]

    # Display the first image and mask from the training dataset
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(X_train[0])
    plt.title("Training Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(y_train[0], cmap='gray')
    plt.title("Training Mask")
    plt.axis('off')
    plt.savefig("UNet_images/Sample Images.png")
    plt.show()

    
    plt.figure(figsize=(10, 5))
    labels = ['Train Images', 'Test Images', 'Train Masks', 'Test Masks']
    counts = [train_images, test_images, train_masks, test_masks]
    colors = ['blue', 'red', 'blue', 'red']
    # add corresponding count on top of bars
    for i in range(len(labels)):
        plt.text(i, counts[i] + 50, str(counts[i]), ha='center', va='bottom')

    plt.bar(labels, counts, color=colors)
    plt.xlabel('Data Type')
    plt.ylabel('Number of Images/Masks')
    plt.title('Number of Images/Masks in Train and Test')
    plt.savefig("UNet_images/Image Breakdown.png")
    plt.show()


    

# construct a recreated version of U-Net
def construct_model():

    inputs = Input(shape=(256, 256, 3))
    
    # Contracting Path
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    # Bottom
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    # Expansive Path
    up6 = Conv2DTranspose(512, 2, strides=(2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(drop5)
    merge6 = Concatenate(axis=3)([drop4, up6])
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2DTranspose(256, 2, strides=(2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
    merge7 = Concatenate(axis=3)([conv3, up7])
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2DTranspose(128, 2, strides=(2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
    merge8 = Concatenate(axis=3)([conv2, up8])
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2DTranspose(64, 2, strides=(2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    merge9 = Concatenate(axis=3)([conv1, up9])
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)

    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=conv10)

    model.summary()

    return model

def construct_small_model():
    inputs = Input(shape=(256, 256, 3))
    
    # Contracting Path
    conv1 = Conv2D(32, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, 3, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, 3, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, 3, activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    # Bottom
    conv5 = Conv2D(512, 3, activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, 3, activation='relu', padding='same')(conv5)

    # Expansive Path
    up6 = Conv2DTranspose(256, 2, strides=(2, 2), activation='relu', padding='same')(conv5)
    merge6 = Concatenate(axis=3)([conv4, up6])
    conv6 = Conv2D(256, 3, activation='relu', padding='same')(merge6)
    conv6 = Conv2D(256, 3, activation='relu', padding='same')(conv6)

    up7 = Conv2DTranspose(128, 2, strides=(2, 2), activation='relu', padding='same')(conv6)
    merge7 = Concatenate(axis=3)([conv3, up7])
    conv7 = Conv2D(128, 3, activation='relu', padding='same')(merge7)
    conv7 = Conv2D(128, 3, activation='relu', padding='same')(conv7)

    up8 = Conv2DTranspose(64, 2, strides=(2, 2), activation='relu', padding='same')(conv7)
    merge8 = Concatenate(axis=3)([conv2, up8])
    conv8 = Conv2D(64, 3, activation='relu', padding='same')(merge8)
    conv8 = Conv2D(64, 3, activation='relu', padding='same')(conv8)

    up9 = Conv2DTranspose(32, 2, strides=(2, 2), activation='relu', padding='same')(conv8)
    merge9 = Concatenate(axis=3)([conv1, up9])
    conv9 = Conv2D(32, 3, activation='relu', padding='same')(merge9)
    conv9 = Conv2D(32, 3, activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=conv10)
    model.summary()

    return model



def compile_and_fit(model, X_train, y_train, X_test, y_test, optimizer='adam', 
                    loss='binary_crossentropy', 
                    metrics=['accuracy',  kb.metrics.Precision(), kb.metrics.Recall(), kb.metrics.BinaryIoU(target_class_ids=[0,1], threshold=0.5)], 
                    epochs=50, batch_size=32):
    # Compile the model
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    model.save('glasses_seg.h5')

    # Train the model
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size)

    # Evaluate the model
    results = model.evaluate(X_test, y_test)

    #quick table of metrics
    for name, value in zip(model.metrics_names, results):
        print(name, ':', value)

    return history



'''two functions to plot the loss and accuracy of the models'''
def plot_loss(res):
    plt.figure(figsize=(10, 10))
    plt.plot(res.history["loss"], label = "train loss")
    plt.plot(res.history["val_loss"], label = "validation loss")
    plt.title("Binary Cross Entropy per Epoch")
    plt.xlabel("Epochs passed")
    plt.ylabel("Binary Cross Entropy")
    plt.legend()
    plt.savefig("UNet_images/lossPerEpoch.png")


def plot_accuracy(res):
    plt.figure(figsize=(10, 10))
    plt.plot(res.history["accuracy"], label = "train accuracy")
    plt.plot(res.history["val_accuracy"], label = "validation accuracy")
    plt.title("Accuracy per Epoch")
    plt.xlabel("Epochs passed")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("UNet_images/accuracyPerEpoch.png")

def plot_iou(res):
    plt.figure(figsize=(10, 10))
    plt.plot(res.history["binary_io_u"], label="train binary IoU")
    plt.plot(res.history["val_binary_io_u"], label="validation binary IoU")
    plt.title("Binary IoU per Epoch")
    plt.xlabel("Epochs passed")
    plt.ylabel("Binary IoU")
    plt.legend()
    plt.savefig("UNet_images/iouPerEpoch.png")


def visualize_results(model, X_test, y_test, num_samples=5):
    # Select random samples from the test set
    sample_indices = np.random.choice(len(X_test), num_samples, replace=False)

    # Make predictions on the test set
    predicted_masks = model.predict(X_test)

    # Plot the results
    plt.figure(figsize=(15, 5 * num_samples))
    for i, idx in enumerate(sample_indices, 1):
        plt.subplot(num_samples, 3, 3 * i - 2)
        plt.imshow(X_test[idx])
        plt.title("Original Image")
        plt.axis('off')

        plt.subplot(num_samples, 3, 3 * i - 1)
        plt.imshow(y_test[idx], cmap='gray')
        plt.title("Real Mask")
        plt.axis('off')

        plt.subplot(num_samples, 3, 3 * i)
        plt.imshow(predicted_masks[idx], cmap='gray')
        plt.title("Predicted Mask")
        plt.axis('off')

    plt.tight_layout()
    plt.savefig("UNet_images/generatedMasksSMALL.png")
    plt.show()

    

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = data_processing()
    # eda(X_train, y_train, X_test, y_test)
    unet = construct_small_model()
    history = compile_and_fit(unet, X_train, y_train, X_test, y_test)
    visualize_results(unet, X_test, y_test)
    # plot_loss(history)
    # plot_accuracy(history)
    # plot_iou(history)


'''
useful links 

https://keras.io/api/metrics/segmentation_metrics/
https://keras.io/examples/vision/grad_cam/
https://keras.io/api/metrics/segmentation_metrics/


'''
