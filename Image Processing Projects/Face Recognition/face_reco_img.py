#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[9]:


keanu = cv2.imread('images/keanu_reeves.jpg',0)
solvay = cv2.imread('images/solvay_conference.jpg',0)


# In[10]:


plt.imshow(keanu, cmap = 'gray')


# In[11]:


plt.imshow(solvay, cmap = 'gray')


# In[14]:


# using haarcascade classifier to detect faces from images
face_cascade = cv2.CascadeClassifier('images/haarcascade_frontalface_default.xml')


# In[19]:


def detect_face(img):
    face_img = img.copy()
    
    face_rects = face_cascade.detectMultiScale(face_img)
    
    for (x,y,w,h) in face_rects:
        cv2.rectangle(face_img, (x,y), (x+w, y+h), (255,255,255), 10)
        
    return face_img


# In[22]:


result = detect_face(solvay)


# In[23]:


plt.imshow(result, cmap = 'gray')


# In[34]:


result = detect_face(keanu)


# In[36]:


plt.imshow(result, cmap = 'gray')


# In[27]:


# using haarcascade classifier to detects eyes from images
eye_cascade = cv2.CascadeClassifier('images/haarcascade_eye.xml')


# In[42]:


def detect_eyes(img):
    face_img = img.copy()
    
    eyes_rects = eye_cascade.detectMultiScale(face_img, scaleFactor = 1.2, minNeighbors = 5)
    
    for (x,y,w,h) in eyes_rects:
        cv2.rectangle(face_img, (x,y), (x+w, y+h), (255,255,255), 10)
        
    return face_img


# In[43]:


result = detect_eyes(keanu)


# In[44]:


plt.imshow(result, cmap = 'gray')


# In[45]:


# capturing face from video
cap = cv2.VideoCapture(0)

while True:
    
    ret,frame = cap.read(0)
    frame = detect_face(frame)
    
    cv2.imshow('Video face detect', frame)
    
    k = cv2.waitKey(1)
    
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()

