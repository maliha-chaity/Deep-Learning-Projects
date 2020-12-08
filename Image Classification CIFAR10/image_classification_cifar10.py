#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Load the dataset from keras datasets module

from keras.datasets import cifar10
import matplotlib.pyplot as plt
 
(train_X,train_Y),(test_X,test_Y)=cifar10.load_data()


# In[3]:


# Plot some images from the dataset to visualize the dataset

n=6
plt.figure(figsize=(20,10))
for i in range(n):
    plt.subplot(330+1+i)
    plt.imshow(train_X[i])
    plt.show()


# In[4]:


# Import the required layers and modules to create our convolution neural net architecture

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils


# In[5]:


# Convert the pixel values of the dataset to float type and then normalize the dataset

train_x=train_X.astype('float32')
test_X=test_X.astype('float32')
 
train_X=train_X/255.0
test_X=test_X/255.0


# In[6]:


# performing the one-hot encoding for target classes

train_Y=np_utils.to_categorical(train_Y)
test_Y=np_utils.to_categorical(test_Y)
 
num_classes=test_Y.shape[1]


# In[7]:


# Creating the sequential model and add the layers

model=Sequential()
model.add(Conv2D(32,(3,3),input_shape=(32,32,3), padding='same',activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Conv2D(32,(3,3),activation='relu',padding='same',kernel_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(512,activation='relu',kernel_constraint=maxnorm(3)))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))


# In[9]:


# Configuring the optimizer and compile the model

sgd=SGD(lr=0.01,momentum=0.9,decay=(0.01/25),nesterov=False)
 
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])


# In[10]:


# Viewing the model summary for better understanding of model architecture

model.summary()


# In[11]:


# Training the model

model.fit(train_X,train_Y, validation_data=(test_X,test_Y), epochs=25,batch_size=32)


# In[13]:


# Calculating its accuracy on testing data

acc=model.evaluate(test_X,test_Y)
print("\n")
print(acc*100)


# In[34]:


# Making a dictionary to map to the output classes and making predictions from the model

results={
   0:'aeroplane',
   1:'automobile',
   2:'bird',
   3:'cat',
   4:'deer',
   5:'dog',
   6:'frog',
   7:'horse',
   8:'ship',
   9:'truck'
}
from PIL import Image
import numpy as np
im=Image.open(r"C:\Users\HP\Desktop\Deep Learning\Deep Learning\Image Classification CIFAR10\Single Prediction\14.jpg")
# the input image is required to be in the shape of dataset, i.e (32,32,3)
 
im=im.resize((32,32))
im=np.expand_dims(im,axis=0)
im=np.array(im)
pred=model.predict_classes([im])[0]
print(pred,results[pred])

