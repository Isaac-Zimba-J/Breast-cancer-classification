# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 12:16:55 2023

@author: Zaac
"""

import numpy as np 
import cv2
import pandas as pd
import os
from matplotlib import pyplot as plt




# use the path in to where your files are just the images without the masks
img_path = "C:/Users/Zaac/Desktop/Workspace/python/Tensorflow/Breast cancer/Dataset/images"

image_dataset = pd.DataFrame() 

########################

# Iterate through all subfolders within the main image directory
for class_folder in os.listdir(img_path):
    class_folder_path = os.path.join(img_path, class_folder)
    
    if not os.path.isdir(class_folder_path):
        continue  # Skip if not a directory
    
    # Iterate through images within the class folder
    for image in os.listdir(class_folder_path):
        print(f" {image} ")
        
        data_frame = pd.DataFrame()
        
        input_image = cv2.imread(os.path.join(class_folder_path, image))
        
####### #################

# for image in os.listdir(img_path):
#     print(image)
    
#     data_frame = pd.DataFrame()
    
        input_imgage = cv2.imread(img_path + image)
        
        if input_imgage.ndim == 3 and input_imgage.shape[-1] == 3:
            img = cv2.cvtColor(input_imgage, cv2.COLOR_BGR2GRAY)
        elif input_imgage.ndim == 2:
            img = input_imgage
        else:
            raise Exception("The module only works for greyscal and RGB values")
            
    #############################################################################
    
        pixel_values = img.reshape(-1)
        data_frame['Pixel_value'] = pixel_values
        data_frame['Image_name'] = image
        
               
    #############################################################################
        num = 1
        kernals = []
        for theta in range(2):
            theta = theta / 4.*np.pi 
            for sigma in range(1,3):
                for lamda in np.arange(0, np.pi, np.pi/4):
                    for gamma in (0.05, 0.5):
                        
                        gabor_label = "Garbor"+ str(num)
                        Kernal_size = 9 
                       
                        kernel = cv2.getGaborKernel((Kernal_size, Kernal_size), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F)  
                        kernals.append(kernel)
                        #
                        filtered_img = cv2.filter2D(img, cv2.CV_8UC3, kernel)
                        filtered_img2 = filtered_img.reshape(-1)
                        data_frame[gabor_label] = filtered_img2
                        cv2.imwrite('C:/Users/Zaac/Desktop/Workspace/python/Tensorflow/Breast cancer/Dataset/train/filters/'+gabor_label+'.png', filtered_img2.reshape(img.shape[:2]))
                       # plt.imsave('C:/Users/Zaac/Desktop/Workspace/python/Tensorflow/Breast cancer/Dataset/train/filters/'+gabor_label+'.png', filtered_img2.reshape(img.shape[:2]), cmap='gray')
    
                        print(gabor_label, ': theta=', theta, ': sigma=', sigma, ': lamda=', lamda, ': gamma=', gamma)
                        num += 1  #Increment for gabor column label
                        
        
                        
        #CANNY EDGE
        edges = cv2.Canny(img, 100,200)   #Image, min and max values
        edges1 = edges.reshape(-1)
        data_frame['Canny Edge'] = edges1 #Add column to original dataframe
      #  cv2.imwrite('C:/Users/Zaac/Desktop/Workspace/python/Tensorflow/Breast cancer/Dataset/train/filters/Canny Edge.png', edges1.reshape(img.shape[:2]))
        
        
        # For color images (e.g., Canny edge)
        plt.imshow(edges1.reshape(img.shape[:2]), cmap='gray')
        plt.title('Canny Edge')
        plt.show()
        
    
        
        from skimage.filters import roberts, sobel, scharr, prewitt
        
        #ROBERTS EDGE
        edge_roberts = roberts(img)
        edge_roberts1 = edge_roberts.reshape(-1)
        data_frame['Roberts'] = edge_roberts1
        #cv2.imwrite('C:/Users/Zaac/Desktop/Workspace/python/Tensorflow/Breast cancer/Dataset/train/filters/Roberts.png', edge_roberts1.reshape(img.shape[:2]))
        # For color images (e.g., Canny edge)
        plt.imshow(edge_roberts1.reshape(img.shape[:2]), cmap='gray')
        plt.title('Roberts')
        plt.show()
        
    
        
        #SOBEL
        edge_sobel = sobel(img)
        edge_sobel1 = edge_sobel.reshape(-1)
        data_frame['Sobel'] = edge_sobel1
        cv2.imwrite('C:/Users/Zaac/Desktop/Workspace/python/Tensorflow/Breast cancer/Dataset/train/filters/Sobel.png', edge_sobel1.reshape(img.shape[:2]))
        
        
        #SCHARR
        edge_scharr = scharr(img)
        edge_scharr1 = edge_scharr.reshape(-1)
        data_frame['Scharr'] = edge_scharr1
        cv2.imwrite('C:/Users/Zaac/Desktop/Workspace/python/Tensorflow/Breast cancer/Dataset/train/filters/Scharr.png', edge_scharr1.reshape(img.shape[:2]))
        
        
        #PREWITT
        edge_prewitt = prewitt(img)
        edge_prewitt1 = edge_prewitt.reshape(-1)
        data_frame['Prewitt'] = edge_prewitt1
        cv2.imwrite('C:/Users/Zaac/Desktop/Workspace/python/Tensorflow/Breast cancer/Dataset/train/filters/Prewitt.png', edge_prewitt1.reshape(img.shape[:2]))
        
        
        #GAUSSIAN with sigma=3
        from scipy import ndimage as nd
        gaussian_img = nd.gaussian_filter(img, sigma=3)
        gaussian_img1 = gaussian_img.reshape(-1)
        data_frame['Gaussian s3'] = gaussian_img1
        cv2.imwrite('C:/Users/Zaac/Desktop/Workspace/python/Tensorflow/Breast cancer/Dataset/train/filters/Gausian sigma 3.png', gaussian_img1.reshape(img.shape[:2]))
        
        
        #GAUSSIAN with sigma=7
        gaussian_img2 = nd.gaussian_filter(img, sigma=7)
        gaussian_img3 = gaussian_img2.reshape(-1)
        data_frame['Gaussian s7'] = gaussian_img3
        cv2.imwrite('C:/Users/Zaac/Desktop/Workspace/python/Tensorflow/Breast cancer/Dataset/train/filters/Gausian sigma 7.png', gaussian_img3.reshape(img.shape[:2]))
        
        
        #MEDIAN with sigma=3
        median_img = nd.median_filter(img, size=3)
        median_img1 = median_img.reshape(-1)
        data_frame['Median s3'] = median_img1
        cv2.imwrite('C:/Users/Zaac/Desktop/Workspace/python/Tensorflow/Breast cancer/Dataset/train/filters/Median S3.png', median_img1.reshape(img.shape[:2]))
        
        # For color images (e.g., Canny edge)
        plt.imshow(median_img1.reshape(img.shape[:2]), cmap='gray')
        plt.title('Median s3')
        plt.show()
                  
    
        #VARIANCE with size=3..*
    #    variance_img = nd.generic_filter(img, np.var, size=3)
    #    variance_img1 = variance_img.reshape(-1)
    #    df['Variance s3'] = variance_img1  #Add column to original dataframe
    
    
    ######################################                    
    #Update dataframe for images to include details for each image in the loop
        #img_dataset = img_dataset.append(data_frame)
        image_dataset = pd.concat([image_dataset, data_frame], ignore_index=True)
    




######################
# out all masks in folders as there sources are example put masks for benighn in a
#  benignn mask folder so on for all 

mask_dataset = pd.DataFrame()
mask_path = "C:/Users/Zaac/Desktop/Workspace/python/Tensorflow/Breast cancer/Dataset/train_masks/"




################################


# Iterate through all subfolders within the main image directory
for mask in os.listdir(mask_path):
    mask_folder_path = os.path.join(mask_path, class_folder)
    
    if not os.path.isdir(mask_folder_path):
        continue  # Skip if not a directory
    
    # Iterate through images within the class folder
    for image in os.listdir(mask_folder_path):
        print(f" {image} ")
        
        data_frame_mask = pd.DataFrame()
        
        input_mask = cv2.imread(os.path.join(mask_folder_path, image))
        

###############################

# for mask in os.listdir(mask_path):
#     print(mask)
    
#     data_frame_mask = pd.DataFrame()
        # input_mask = cv2.imread(mask_path + mask)
        
        if input_mask.ndim == 3 and input_mask.shape[-1] == 3:
            label = cv2.cvtColor(input_mask, cv2.COLOR_BGR2GRAY)
        elif input_mask.ndim == 2:
            label = input_mask
        else:
            raise Exception("The module only works for greyscal and RGB values")
    
        label_values = label.reshape(-1)
        data_frame_mask['Label_value'] = label_values
        data_frame_mask['Mask_name'] = mask
    
        #mask_dataset = mask_dataset.append(data_frame_mask)
        mask_dataset = pd.concat([mask_dataset, data_frame_mask], ignore_index=True)
       # img_dataset = pd.concat([img_dataset, data_frame], ignore_index=True)

dataset = pd.concat([image_dataset, mask_dataset], axis=1)    #Concatenate both image and mask datasets

dataset = dataset[dataset.Label_value != 0]

# Save the dataset to a CSV file
csv_filename = "image_features_dataset.csv"
dataset.to_csv(csv_filename, index=False)

print(f"Dataset saved to {csv_filename}")
