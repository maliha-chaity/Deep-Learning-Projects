#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


road = cv2.imread('images/nature.jpg')


# In[3]:


road_copy = road.copy()


# In[4]:


plt.imshow(road)


# In[5]:


road.shape


# In[6]:


road.shape[:2]


# In[7]:


marker_image = np.zeros(road.shape[:2], dtype = np.int32)


# In[8]:


segments = np.zeros(road.shape, dtype = np.uint8)


# In[9]:


marker_image.shape


# In[10]:


segments.shape


# In[11]:


from matplotlib import cm


# In[12]:


cm.tab10(0)


# In[13]:


def create_rgb(i):
    return tuple(np.array(cm.tab10(i)[:3])*255)


# In[14]:


colors = []
for i in range(10):
    colors.append(create_rgb(i))


# In[15]:


colors


# In[16]:


n_markers = 10
current_marker = 1
marks_updated = False


# In[17]:


def mouse_callback(event, x, y, flags, param):
    global marks_updated
    
    if event == cv2.EVENT_LBUTTONDOWN:
        # markers passed to the watershed algo
        cv2.circle(marker_image, (x,y), 10, (current_marker), -1)
        
        # user sees on the road image
        cv2.circle(road_copy, (x,y), 10, colors[current_marker], -1)
        
        marks_updated = True


# In[19]:


cv2.namedWindow('Road Image')
cv2.setMouseCallback('Road Image', mouse_callback)

while True:
    cv2.imshow('Watershed Segments', segments)
    cv2.imshow('Road Image', road_copy)
    
    
    # close all windows
    k = cv2.waitKey(1)
    
    if k == 27:
        break
        
    if k == ord('c'):
        road_copy = road.copy()
        marker_image = np.zeros(road.shape[:2], dtype = np.int32)
        segments = np.zeros(road.shape, dtype = np.uint8)
                            
    if (k>0) and chr(k).isdigit():
        current_marker = int(chr(k))
                            
    if marks_updated:
        marker_image_copy = marker_image.copy()
        cv2.watershed(road, marker_image_copy)
        
        segments = np.zeros(road.shape, dtype = np.uint8)
                            
        for color_ind in range(n_markers):
            segments[marker_image_copy == (color_ind)] == colors[color_ind]
                            
cv2.destroyAllWindows()


# In[ ]:




