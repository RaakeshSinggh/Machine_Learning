#!/usr/bin/env python
# coding: utf-8

# In[49]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[50]:


from keras.datasets import mnist


# In[53]:


(x_train, y_train), (x_test, y_test) = mnist.load_data()


# In[52]:


x_train.shape, y_train.shape, x_test.shape, y_test.shape


# In[54]:


x_train[0]


# In[55]:


x_train[0].shape


# In[56]:


len(x_train[0]), len(x_train[0][0])


# In[57]:


plt.imshow(x_train[0])


# In[66]:


y_train[0]


# In[67]:


28*28


# In[69]:


dummy = pd.DataFrame(x_train[0:2])


# In[75]:


dummy['label'] = y_train[0]


# In[76]:


dummy


# In[77]:


for i in range(4):
  plt.imshow(x_train[i])
  plt.show()


# In[78]:


y_train[3]


# In[79]:


for i in range(4):
  plt.imshow(x_train[i], cmap = plt.get_cmap('gray'))
  plt.show()


# In[80]:


x_train.shape


# In[81]:


x_train = x_train.reshape((x_train.shape[0],28,28,1))


# In[82]:


x_train.shape


# In[83]:


x_train[0]


# In[84]:


x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))


# In[85]:


np.unique(y_train)


# In[86]:


y_train


# In[87]:


from keras.utils import to_categorical


# In[88]:


y_train = to_categorical(y_train)


# In[89]:


y_train[0]


# In[90]:


y_test = to_categorical(y_test)


# In[93]:


from __future__ import print_function


# In[97]:


from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPool2D


# In[98]:


model = Sequential()
model.add(Conv2D(32, (3, 3), activation="relu", input_shape = (28, 28, 1)))
model.add(MaxPool2D(2, 2))
model.add(Flatten())
model.add(Dense(64, activation = 'relu'))
model.add(Dense(10, activation = 'softmax'))


# In[99]:


model.summary()


# In[100]:


346176 + 650 + 320


# In[101]:


model.compile(optimizer = 'adam', loss='categorical_crossentropy', 
              metrics=['accuracy'])


# In[102]:


model.fit(x_train, y_train, epochs = 10, batch_size = 32)


# In[103]:


_, acc = model.evaluate(x_test, y_test)


# In[104]:


acc


# In[105]:


y_pred = model.predict(x_test)


# In[106]:


for i, j in zip(y_test[:3], y_pred[:3]):
  print(i, np.argmax(j))
  print('-------------')

