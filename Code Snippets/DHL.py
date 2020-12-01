import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
%matplotlib inline

import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
import itertools

from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Dropout, concatenate, Input, Conv2D, MaxPooling2D
from keras.optimizers import Adam, Adadelta
from keras.layers.advanced_activations import LeakyReLU
from keras.utils.np_utils import to_categorical
from sklearn.metrics import confusion_matrix



def fire_incept(x, fire=16, intercept=64):
    '''
    Defining Incept Layer
    '''
    x = Conv2D(fire, (5,5), strides=(2,2))(x)
    x = LeakyReLU(alpha=0.15)(x)
    
    left = Conv2D(intercept, (3,3), padding='same')(x)
    left = LeakyReLU(alpha=0.15)(left)
    
    right = Conv2D(intercept, (5,5), padding='same')(x)
    right = LeakyReLU(alpha=0.15)(right)
    
    x = concatenate([left, right], axis=3)
    return x
 
def fire_squeeze(x, fire=16, intercept=64):
    '''
    Defining Squeeze Layer
    '''
    x = Conv2D(fire, (1,1))(x)
    x = LeakyReLU(alpha=0.15)(x)
    
    left = Conv2D(intercept, (1,1))(x)
    left = LeakyReLU(alpha=0.15)(left)
    
    right = Conv2D(intercept, (3,3), padding='same')(x)
    right = LeakyReLU(alpha=0.15)(right)
    
    x = concatenate([left, right], axis=3)
    return x
 
image_input=Input(shape=input_shape)
 
x = fire_incept((image_input), fire=16, intercept=16)
 
x = fire_incept(x, fire=32, intercept=32)
x = fire_squeeze(x, fire=32, intercept=32)
 
x = fire_incept(x, fire=64, intercept=64)
x = fire_squeeze(x, fire=64, intercept=64)
 
x = fire_incept(x, fire=64, intercept=64)
x = fire_squeeze(x, fire=64, intercept=64)
 
x = Conv2D(64, (3,3))(x)
x = LeakyReLU(alpha=0.1)(x)
 
x = Flatten()(x)
 
x = Dense(512)(x)
x = LeakyReLU(alpha=0.1)(x)
x = Dropout(0.1)(x)
 
out = Dense(2, activation='softmax')(x)
 
model_new = Model(image_input, out)
model_new.summary()

import tensorflow as tf
from sklearn.metrics import roc_auc_score

model_new.compile(optimizer = Adam(lr=.00025) , loss = 'categorical_crossentropy', metrics=['accuracy'])

from keras.callbacks import ModelCheckpoint
# learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, 
#                                             factor=0.5, min_lr=0.00001)
filepath="weights-improvement-{epoch:02d}-{val_auc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_auc', verbose=1, save_best_only=True, mode='max')
# callbacks_list = [checkpoint]
# Adding Early Stopping
early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=3)

datagen = ImageDataGenerator(rotation_range=40, zoom_range = 0.2, width_shift_range=0.2, height_shift_range=0.2,
                             horizontal_flip=True, vertical_flip=True)
datagen.fit(x_train)

batch_size = 32
epochs = 250

history = model_new.fit_generator(datagen.flow(x_train_res,y_train_res, batch_size=batch_size), epochs = epochs,
                                  validation_data = (x_val,y_val), verbose = 1, 
                                  steps_per_epoch=x_train.shape[0] // batch_size, 
                                  callbacks=[checkpoint])
                                  
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import seaborn as sns

def evaluatation_metrics(y_true, y_pred,model):
    '''
    Model Evaluation Metric Module
    '''
    accuracy = accuracy_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred, average='weighted')
    cm = confusion_matrix(y_true, y_pred)

    print("Accuracy of",model,": {:.2f}".format(accuracy))
    print("ROC AUC Score of", model,": {:.2f}".format(roc_auc))
    print("Confusion Matrix of", model,": \n",cm)
    
    plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues');
    plt.ylabel('Actual label');
    plt.xlabel('Predicted label');
    title = 'AUC-ROC Score: {:.2f}'.format(roc_auc)
    plt.title(title)
    plt.savefig(f"CM_{model}.png")
    plt.show()
    
    
# DHL Late Fusion -- extracting the features
model_feat = Model(inputs=model_new.input,outputs=model_new.get_layer('dense').output)
feat_train = model_feat.predict(x_train_res)
print(feat_train.shape)

feat_val = model_feat.predict(x_val)
print(feat_val.shape)
    
# Applying DHL + XGBoost
import xgboost as xgb
xb = xgb.XGBClassifier()
xb.fit(feat_train,np.argmax(y_train_res,axis=1))

# Applying DHL + Random Forest
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(max_depth=4, random_state=0)
clf.fit(feat_train,np.argmax(y_train_res,axis=1))
