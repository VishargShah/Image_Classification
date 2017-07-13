# -*- coding: utf-8 -*-
"""
Created on Tue May  2 18:18:58 2017

@author: Visharg Shah
"""

#Importing the keras lib and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#Initializing the CNN
classifier = Sequential()

#Step 1 Convolution
classifier.add(Convolution2D(32,3,3,input_shape = (64,64,3),activation = 'relu'))

#Step 2 Pooliong
classifier.add(MaxPooling2D(pool_size = (2,2)))

#Adding the second Convolutional layer
classifier.add(Convolution2D(32,3,3,activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))

#Step 3 Flattening
classifier.add(Flatten())

#Step 4 Full Connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

#Compile the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Fitting the CNN to the Images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
                                   rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64,64), batch_size = 32, 
                                                    class_mode = 'binary')

test_set = train_datagen.flow_from_directory('dataset/test_set',
                                                 target_size = (64,64), batch_size = 32, 
                                                    class_mode = 'binary')

classifier.fit_generator(training_set,
                         samples_per_epoch = 8000,
                         nb_epoch = 25,
                         validation_data = test_set,
                         nb_val_samples = 2000)
