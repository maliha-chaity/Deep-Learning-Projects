#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


img = cv2.imread('images/car_number.jpg')


# In[3]:


# reshaping the image and correct coloring for matplotlib
def display(img):
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111)
    new_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax.imshow(new_img)


# In[4]:


display(img)


# In[5]:


plate_cascade = cv2.CascadeClassifier('images/haarcascade_russian_plate_number.xml')


# In[9]:


# detecting plate number of cars from image

def detect_plate(img):
    plate_img = img.copy()
    
    plate_rects = plate_cascade.detectMultiScale(plate_img, scaleFactor = 1.3, minNeighbors = 3)
    
    for (x,y,w,h) in plate_rects:
        cv2.rectangle(plate_img, (x,y), (x+w, y+h), (0,0,255), 4)
        
    return plate_img


# In[10]:


result = detect_plate(img)


# In[11]:


display(result)

