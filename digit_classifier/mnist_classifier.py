#Imports
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D


#Load dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

#Reshaping and Normalizing the Images
# Reshaping the array to 4-dims so that it can work with the Keras API
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)
# Making sure that the values are float so that we can get decimal points after division
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# Normalizing the RGB codes by dividing it to the max RGB value.
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print('Number of images in x_train', x_train.shape[0])
print('Number of images in x_test', x_test.shape[0])

#Model part
# Creating a Sequential Model and adding the layers
model = Sequential()
model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
model.add(Dense(128, activation=tf.nn.relu))
model.add(Dropout(0.2))
model.add(Dense(10,activation=tf.nn.softmax))

#fit the model
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])
model.fit(x=x_train,y=y_train, epochs=1)
model.summary()

#Evaluate
#model.evaluate(x_test, y_test)
#predict
image_index = 3333
plt.imshow(x_test[image_index].reshape(28, 28),cmap='Greys')
#print("image")
#print(x_test[image_index])
pred = model.predict(x_test[image_index].reshape(1, 28, 28, 1))
#print("after reshaping")
#print(x_test[image_index].reshape(1, 28, 28, 1))
print(pred.argmax())
#check
import cv2
im = cv2.imread("./test/one.jpeg")
img = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
print(im.shape)
#im.reshape(1, 28, 28, 1)
res = cv2.resize(img, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)
#print(im)
#img_gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
res.reshape(1, 28, 28, 1)
cv2.imshow('ImageWindow', res)
cv2.waitKey(5000)

print(res.shape)
pred1 = model.predict(res.reshape(1, 28, 28, 1))
print("predicted value")
print(pred1.argmax())
