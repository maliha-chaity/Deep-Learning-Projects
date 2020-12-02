#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


def display(img, cmap = 'gray'):
    fig = plt.figure(figsize = (12,10))
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap = 'gray')


# In[98]:


sep_coins = cv2.imread('images/coins3.jpg')


# In[99]:


display(sep_coins)


# In[100]:


# at first, apply median blur to the image
sep_blur = cv2.medianBlur(sep_coins, 25)


# In[101]:


display(sep_blur)


# In[102]:


# convert it to gray scale
gray_sep_coins = cv2.cvtColor(sep_blur, cv2.COLOR_BGR2GRAY)


# In[103]:


display(gray_sep_coins)


# In[104]:


# apply binary threshold in order to make it black and white and seperate it according to foreground and background
ret, sep_thresh = cv2.threshold(gray_sep_coins, 245, 255, cv2.THRESH_BINARY_INV)


# In[105]:


display(sep_thresh)


# In[106]:


# find contours
image, contours, hierarchy = cv2.findContours(sep_thresh.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)


# In[108]:


for i in range(len(contours)):
    if hierarchy[0][i][3] == -1:
        cv2.drawContours(sep_coins, contours, i, (255,0,0), 10)


# In[109]:


display(sep_coins)


# In[110]:


img = cv2.imread('images/coins3.jpg')


# In[111]:


# blur the image
img = cv2.medianBlur(img, 35)


# In[112]:


display(img)


# In[113]:


# convert the image into gray scale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# In[116]:


# apply a threshold on the gray scale image
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)


# In[117]:


display(thresh)


# In[120]:


# noise removal (if there is any). We can use OTSU thresholding or follow the following approach in order to remove noise.
kernel = np.ones((3,3),np.uint8)


# In[121]:


kernel


# In[123]:


opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations = 2)


# In[124]:


display(opening)


# In[125]:


# distance transformation
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2,5)


# In[126]:


display(dist_transform)


# In[130]:


sure_bg = cv2.dilate(opening, kernel, iterations = 3)


# In[131]:


display(sure_bg)


# In[127]:


# applying further threshold to identify objects on the foreground

ret, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)


# In[128]:


display(sure_fg)


# In[132]:


# find and display the unknown region

sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)


# In[133]:


display(unknown)


# In[134]:


# create label markers for watershed algorithm

ret, markers = cv2.connectedComponents(sure_fg)


# In[136]:


markers = markers + 1


# In[137]:


markers[unknown == 255] = 0


# In[138]:


display(markers)


# In[139]:


markers = cv2.watershed(img, markers)


# In[140]:


display(markers)


# In[141]:


# find contours
image, contours, hierarchy = cv2.findContours(markers.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)


# In[146]:


for i in range(len(contours)):
    if hierarchy[0][i][3] == -1:
        cv2.drawContours(sep_coins, contours, i, (255,0,0), 2)


# In[147]:


display(sep_coins)

