# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 21:48:39 2023

@author: Zaac
"""

import os
import sys
import tkinter as tk
import customtkinter
import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageFont, ImageDraw
import shutil
import numpy as np
import cv2
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage.filters import sobel, scharr, prewitt, roberts


from scipy import ndimage as nd

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS2
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)



# Specify the desktop path
desktop_path = os.path.expanduser("~/Desktop")

# Define folder names
folders = ["breast_images", "Extracted", "Segmentated"]

# Create folders on the desktop
for folder in folders:
    folder_path = os.path.join(desktop_path, folder)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Created folder on desktop: {folder_path}")
    else:
        print(f"Folder already exists on desktop: {folder_path}")

# Update destination folder paths
destination_folder = os.path.join(desktop_path, "breast_images")
destination_folder_segmented = os.path.join(desktop_path, "Segmentated")
destination_folder_extracted = os.path.join(desktop_path, "Extracted")
# ... (Rest of the import statements)
# Define a function to extract Gabor features
def Gabor_extract(input_image):
    num = 1
    features = []
    for theta in range(2):
        theta = theta / 2. * np.pi
        for sigma in range(1, 3):
            for lamda in np.arange(3, np.pi, np.pi / 4):
                for gamma in (0.05, 0.5):
                    kernel_size = 3
                    kernel = cv2.getGaborKernel((kernel_size, kernel_size), sigma, theta, lamda, gamma, 0,
                                                ktype=cv2.CV_32F)
                    filtered_img = cv2.filter2D(input_image, cv2.CV_8UC3, kernel)
                    filtered_img2 = filtered_img.reshape(-1)
                    features.append(filtered_img2)
                    num += 1

    edges = cv2.Canny(input_image, 100, 200)
    edges1 = edges.reshape(-1)
    features.append(edges1)

    median_img = cv2.medianBlur(input_image, 3)
    median_img1 = median_img.reshape(-1)
    features.append(median_img1)

    return features

# Define a function to extract features from a single image
def extract_single_image_features(image_path):
    # Create an empty DataFrame to store the features
    data_frame = pd.DataFrame()

    # Load the image
    input_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Feature Extraction: Gabor Filters
    features = Gabor_extract(input_image)

    # Create column names for Gabor features
    gabor_columns = [f'Gabor_{i}' for i in range(len(features))]

    # Assign Gabor features to DataFrame columns
    for col_name, feature in zip(gabor_columns, features):
        data_frame[col_name] = feature
        # Plot the Gabor filter
        plt.figure()
        plt.imshow(feature.reshape(input_image.shape), cmap='gray')
        plt.title(col_name)
        plt.axis('off')

    # CANNY EDGE
    edges = cv2.Canny(input_image, 100, 200)  # Image, min and max values
    edges1 = edges.reshape(-1)
    data_frame['Canny Edge'] = edges1  # Add column to original dataframe
    
    
    #plot canny edge
    plt.figure()
    plt.imshow(edges, cmap='gray')
    plt.title('Canny Edge')
    plt.axis('off')


    # MEDIAN with sigma=3
    median_img = nd.median_filter(input_image, size=3)
    median_img1 = median_img.reshape(-1)
    data_frame['Median s3'] = median_img1
    
    # Plot the Median filter
    plt.figure()
    plt.imshow(median_img, cmap='gray')
    plt.title('Median s3')
    plt.axis('off')


    # Save the DataFrame to a CSV file
    csv_filename = os.path.join(destination_folder_extracted, "single_image_features.csv")
    data_frame.to_csv(csv_filename, index=False)
    
    # Show all the generated plots
    plt.show()

# Create a function to open and process a single image
def process_single_image():
    global img_original
    image_path = filedialog.askopenfilename(title="Select Segmented Image",
                                            initialdir=destination_folder_segmented,
                                            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.gif *.bmp")])

    if image_path:
        img_original = Image.open(image_path)
        img = img_original.resize((400, 300))  # Resize the image to fit the window
        img = ImageTk.PhotoImage(img)

        image_label.config(image=img)
        image_label.image = img
        input_label.config(text="Input Image: " + os.path.basename(image_path))
        dimensions_label.config(
            text="Dimensions: {} x {}".format(img_original.width, img_original.height))

        # Move the image to the destination folder after displaying
        global filename
        filename = os.path.basename(image_path)
        destination_path = os.path.join(destination_folder, filename)
        shutil.move(image_path, destination_path)

        # Extract features from the selected image
        extract_single_image_features(destination_path)
        feature_extraction_label.config(text="Feature Extraction : Success")

def open_image():
    global img_original
    image_path = filedialog.askopenfilename(title="Select Image", filetypes=[("Image Files", "*.png *.jpg *.jpeg *.gif *.bmp")])
    
    if image_path:
        img_original = Image.open(image_path)
        img = img_original.resize((400, 300))  # Resize the image to fit the window
        img = ImageTk.PhotoImage(img)
        
        image_label.config(image=img)
        image_label.image = img
        input_label.config(text="Input Image: " + os.path.basename(image_path))
        dimensions_label.config(text="Dimensions: {} x {}".format(img_original.width, img_original.height))

        # Move the image to the destination folder after displaying
        global filename
        filename = os.path.basename(image_path)
        destination_path = os.path.join(destination_folder, filename)
        shutil.move(image_path, destination_path)

def clear_image():
    image_label.config(image=None)
    input_label.config(text="Input Image:")
    dimensions_label.config(text="Dimensions:")
    segmentation_label.config(text="Segmentation :")
    feature_extraction_label.config(text="Feature Extraction :")
    classification_label.config(text="Confidence : ") 
    class_label.config(text="Class : ") 

def segment_image_fn(model, image):
    # Preprocess the image (resize, normalize, etc.)
    processed_image = image / 255.0  # Normalize to [0, 1]
    processed_image = tf.image.resize(processed_image, (128, 128))
    
    # Perform segmentation
    mask_pred, _ = model.predict(np.expand_dims(processed_image, axis=0))
    segmented_mask = (mask_pred[0] >= 0.5).astype('int32')
    
    return segmented_mask

def segment_image():
    # Load the trained segmentation model
    # loaded_model = tf.keras.models.load_model('models/breast_cancer_segmentation.h5')
    loaded_model = tf.keras.models.load_model(resource_path('models\\breast_cancer_segmentation.h5'))
   

    
    # Load an image you want to segment from the destination folder
    input_image_path = os.path.join(destination_folder, filename)
    input_image = cv2.imread(input_image_path)
    
    # Segment the image
    segmented_mask = segment_image_fn(loaded_model, input_image)
    
    # Convert the binary mask to a grayscale image
    segmented_image = segmented_mask * 255
    
    # Visualize the original image and the segmented mask
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(input_image)
    plt.title('Input Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(segmented_image, cmap='gray')  # Display the grayscale segmented image
    plt.title('Segmented Mask')
    plt.axis('off')
    
    # Save segmented images to the segmented folder
    os.makedirs(destination_folder_segmented, exist_ok=True)
    input_image_path = os.path.join(destination_folder_segmented, "input_{}.png".format(filename))
    mask_image_path = os.path.join(destination_folder_segmented, "mask_{}.png".format(filename))
    
    cv2.imwrite(input_image_path, cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
    cv2.imwrite(mask_image_path, segmented_image)
    
    plt.show()
    segmentation_label.config(text="Segmentation : Succes")
    
    
    
def display_class_and_confidence(image, class_label, confidence):
    img_with_text = image.copy()
    draw = ImageDraw.Draw(img_with_text)
    font = ImageFont.load_default()  # You can choose a different font if desired
    text = f"Class: {class_label}\nConfidence: {confidence:.2f}"
    draw.text((10, 10), text, font=font, fill=(255, 0, 0))  # Red color
    return img_with_text


def classify_image(model, image):
    processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    processed_image = processed_image / 255.0
    processed_image = tf.image.resize(processed_image, (256, 256))

    class_probs = model.predict(np.expand_dims(processed_image, axis=0))[0]
    predicted_class_index = np.argmax(class_probs)
    confidence = class_probs[predicted_class_index]

    class_labels = ['Benign','Normal','Malignant']
    predicted_class_label = class_labels[predicted_class_index]

    return predicted_class_label, confidence


# ... (previous code)

def classify_extracted_image():
    # Load the trained classification model
    # loaded_model = tf.keras.models.load_model('models/classification.h5')
    loaded_model = tf.keras.models.load_model(resource_path('models\\classification.h5'))

    # Select an extracted image to classify
    image_path = filedialog.askopenfilename(title="Select Extracted Image", initialdir=destination_folder_extracted, filetypes=[("Image Files", "*.png")])

    if image_path:
        extracted_image = cv2.imread(image_path)
        if extracted_image is None:
            print(f"Error: Unable to load image from '{image_path}'")
            return
        
        # Classify the extracted image
        predicted_class_label, confidence = classify_image(loaded_model, extracted_image)
       
        classification_label.config(text=f"Confidence : {confidence}") 
        class_label.config(text=f"Class : {predicted_class_label}") 


# Create the main window using customtkinter
root = ctk.CTk()
root.title("Breast cancer classifier")
root.geometry("900x400")  # Fixed window size
root.resizable(False, False)  # Make the window unresizable


# Create a left frame for buttons
button_frame = tk.Frame(root, width=100, bg="lightgray")
button_frame.pack(side=tk.LEFT, fill=tk.Y)



# Create a middle frame for image display
image_frame = tk.Frame(root, width=600)
image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

# Create a right frame for empty space with a different color
empty_frame = tk.Frame(root, width=350, bg="lightgray")
empty_frame.pack(side=tk.RIGHT, fill=tk.Y)

# Create a label widget to display the image
image_label = tk.Label(image_frame)
image_label.pack(padx=10, pady=10)

# Create an "Input Image" label and a "Dimensions" label in the empty frame
input_label = tk.Label(empty_frame,font=("Helvetica", 14), text="Input Image:", bg="lightgray", fg="white")
input_label.pack(padx=10, pady=5, anchor="w")

dimensions_label = tk.Label(empty_frame,font=("Helvetica", 14), text="Dimensions:", bg="lightgray", fg="white")
dimensions_label.pack(padx=10, pady=5, anchor="w")


segmentation_label = tk.Label(empty_frame, font=("Helvetica", 14),text="Segmentation :", bg="lightgray", fg="green")
segmentation_label.pack(padx=10, pady=5, anchor="w")


feature_extraction_label = tk.Label(empty_frame,font=("Helvetica", 14), text="Feature Extraction :", bg="lightgray", fg="red")
feature_extraction_label.pack(padx=10, pady=5, anchor="w")

classification_label = tk.Label(empty_frame, font=("Helvetica", 14),text="Classify :", bg="lightgray", fg="blue")
classification_label.pack(padx=10, pady=5, anchor="w")

class_label = tk.Label(empty_frame, font=("Helvetica", 14),text="Class :", bg="lightgray", fg="blue")
class_label.pack(padx=10, pady=5, anchor="w")

# Create an "Open Image" button
open_button = ctk.CTkButton(button_frame,font=("Helvetica", 14), text="Open Image", command=open_image)
open_button.pack(side=tk.TOP, padx=10, pady=10, fill=tk.X)

# Create three buttons, including a "Segment Image" button

button2 = ctk.CTkButton(button_frame,font=("Helvetica", 14), text="Segment Image", command=segment_image)
button2.pack(side=tk.TOP, padx=10, pady=5, fill=tk.X)

# Create an "Process Single Image" button
process_single_image_button = ctk.CTkButton(button_frame,font=("Helvetica", 14), text="Extract features", command=process_single_image)
process_single_image_button.pack(side=tk.TOP, padx=10, pady=5, fill=tk.X)

# Create a "Classify Image" button
classify_button = ctk.CTkButton(button_frame, font=("Helvetica", 14),text="Classify Image", command=classify_extracted_image)
classify_button.pack(side=tk.TOP, padx=10, pady=5, fill=tk.X)


button3 = ctk.CTkButton(button_frame, font=("Helvetica", 14),text="Clear Image", command=clear_image)
button3.pack(side=tk.TOP, padx=10, pady=5, fill=tk.X)


# Run the Tkinter event loop
root.mainloop()
