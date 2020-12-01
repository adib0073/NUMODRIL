import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

def create_mask_for_image(image):
    '''
    Utility Function to create segmented morphological masks
    '''
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    sensitivity = 35
    lower_hsv = np.array([60 - sensitivity, 100, 50])
    upper_hsv = np.array([60 + sensitivity, 255, 255])

    mask = cv2.inRange(image_hsv, lower_hsv, upper_hsv)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask
 
def segment_image(image):
    '''
    Utility Function to apply segmented morphological masks
    '''
    mask = create_mask_for_image(image)
    output = cv2.bitwise_and(image, image, mask = mask)
    return output
 
def sharpen_image(image):
    '''
    Utility Function to sharpen processed images
    '''
    image_blurred = cv2.GaussianBlur(image, (0, 0), 3)
    image_sharp = cv2.addWeighted(image, 1.5, image_blurred, -0.5, 0)
    return image_sharp
    
    
    
x_train = []

for i in range(len(train)):
    #print(train_df['file'][i])
    img = cv2.imread(train_df['file'][i])
    # cv2_imshow(img)
    # img = cv2.resize(img,dsize=(256,256))
    if i == 5 :
      print('Original Image')
      cv2_imshow(img)
    img_stack = segment_image(img)
    if i == 5 :
      print('Segmented Image')
      cv2_imshow(img_stack)
    img_stack = sharpen_image(img_stack)
    if i == 5 :
      print('Sharpened Image')
      cv2_imshow(img_stack)
    img_stack = cv2.cvtColor( img_stack, cv2.COLOR_RGB2GRAY )
    if i == 5 :
      print('Gray Image')
      cv2_imshow(img_stack)
    img_stack = np.reshape(img_stack,(1024,1024,1))
    x_train.append(np.concatenate((np.array(img),np.array(img_stack)),axis=2))

x_train = np.array(x_train)
   
