
# coding: utf-8

# In[19]:

import cv2
import matplotlib.pyplot as plt


# In[20]:

src = cv2.imread("test1.png", cv2.CV_8UC1)
plt.imshow(src, cmap='Greys_r')


# In[21]:

dist = cv2.distanceTransform(src, cv2.cv.CV_DIST_L2, 5)#v2.cv.CV_DIST_MASK_PRECISE)
plt.imshow(dist, cmap='Greys_r')


# In[22]:

inv = ~src
inv_dist = -cv2.distanceTransform(inv, cv2.cv.CV_DIST_L2, 5)#cv2.cv.CV_DIST_MASK_PRECISE)
plt.imshow(inv_dist, cmap ='Greys_r' )


# In[23]:

merge = dist+inv_dist
plt.imsave("merge.png", merge, cmap=plt.cm.gray)
plt.imshow(merge, cmap ='Greys_r' )


# In[24]:

template = cv2.resize(merge, (64,64))
plt.imsave("template.png", template, cmap=plt.cm.gray)
plt.imshow(template, cmap ='Greys_r')


# In[25]:

import numpy as np


# In[28]:

large = cv2.resize(template, (2048, 2048))
bin = cv2.threshold(large, 0., 255, cv2.THRESH_BINARY)
plt.imshow(bin[1], cmap ='Greys_r')


# In[29]:

plt.imsave("dst.png",bin[1], cmap=plt.cm.gray)


# In[ ]:



