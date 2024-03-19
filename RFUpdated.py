import warnings
import os
import pickle
warnings.filterwarnings('ignore')


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics


from sklearn.ensemble import RandomForestClassifier


from tensorflow.keras.preprocessing.image import load_img, img_to_array


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
        
        # Convert image and mask to arrays and normalize
        img_array = img_to_array(img) / 255.0  
        mask_array = img_to_array(mask) / 255.0 
        
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



def fit(X_train, y_train, X_test, y_test):

    #take 100 samples of train
    X_train = X_train[:100]
    y_train =  y_train[:100].astype(int) 

    print(y_train)

    #take the first 25 samples of test
    X_test = X_test[:25]
    y_test = y_test[:25].astype(int)

    fileForModel = "RFmodel.pickle"

    # Initialize and train the random forest model
    rf_model = RandomForestClassifier(n_estimators=50, n_jobs=-1, verbose=3)
    print("Fitting the model")
    model = rf_model.fit(X_train, y_train)


    pickle.dump(model, open(fileForModel, 'wb'))


    return model


def visualize_results(model, X_test, y_test, num_samples=5):
    # Select random samples from the test set
    sample_indices = np.random.choice(len(X_test[:100]), num_samples, replace=False)

    # Make predictions on the test set
    predicted_masks_flat = model.predict(X_test[:100].reshape(-1, 256*256*3))

    # Reshape the predictions back to original dimensions
    predicted_masks = predicted_masks_flat.reshape(-1, 256, 256)

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
    plt.savefig("RF_images/generatedMasks.png")
    plt.show()


def evaluate_model(model, X_test, y_test):
    # Make predictions on the test set
    y_pred = model.predict(X_test[:50])
    y_pred = y_pred.astype('int')

    y_test = y_test[:50].astype('int')

    y_pred_images = y_pred.reshape(-1, 256, 256)
    y_test_images = y_test.reshape(-1, 256, 256)

    accuracy = np.mean(y_pred_images == y_test_images)

    # calculation for IOU
    intersection = np.logical_and(y_pred, y_test)
    union = np.logical_or(y_pred, y_test)
    iou_score = np.sum(intersection) / np.sum(union)

    print("Pixel-wise Accuracy:", accuracy)
    print("IOU Score:", iou_score)

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = data_processing()

    # #flatten
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    y_train_flat = y_train.reshape(y_train.shape[0], -1)

    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    y_test_flat = y_test.reshape(y_test.shape[0], -1)

    

    randomForest = fit(X_train_flat, y_train_flat, X_test_flat, y_test_flat)

    #load model into randomForest if already saved
    #randomForest = pickle.load(open("RFmodel.pickle", 'rb'))

    evaluate_model(randomForest, X_test_flat, y_test_flat)
    visualize_results(randomForest, X_test, y_test.astype('int'))




