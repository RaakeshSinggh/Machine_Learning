#!/usr/bin/env python
# coding: utf-8

# In[226]:


import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,plot_tree


# In[227]:


data = load_iris()


# In[228]:


data.keys()


# In[230]:


data.feature_names


# In[231]:


data.target_names


# In[232]:


data.target


# In[233]:


data.data_module


# In[234]:


df = pd.DataFrame(data.data,columns=data.feature_names)


# In[235]:


df


# In[236]:


df['target']=y


# In[237]:


df.sample(150)


# In[238]:


x = data['data']
y = data['target']
feature_names=data['feature_names']
features_classes=data['target_names']


# In[239]:


df1=pd.DataFrame(x)
df1['target']=y


# In[240]:


df1['flower_name'] = df1.target.apply(lambda x: data.target_names[x])
df1.head(150)


# In[241]:


df['flower_name'] = df.target.apply(lambda x: data.target_names[x])
df


# In[242]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)


# In[243]:


x_train.shape,x_test.shape,y_train.shape,y_test.shape


# In[244]:


pd.Series(y_train).value_counts()


# In[245]:


model=DecisionTreeClassifier()


# In[273]:


model.fit(x_train,y_train)
model.score(x_test,y_test)


# In[274]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,model.predict(x_test))


# In[316]:


from sklearn.model_selection import cross_val_score,RepeatedStratifiedKFold


# In[317]:


cv = RepeatedStratifiedKFold(n_repeats=3 , random_state=0)


# In[322]:


# score=cross_val_score(model,x_train,y_train)


# In[275]:


from sklearn.model_selection import cross_val_score


# In[276]:


score=cross_val_score(model,x_train,y_train,cv=10)


# In[277]:


score.mean()


# In[278]:


import matplotlib.pyplot as plt
plt.figure(figsize=(20,10))
plot_tree(model,feature_names=feature_names)


# In[296]:


x= list(range(1,21))


# In[297]:


x


# In[298]:


from sklearn.model_selection import KFold


# In[299]:


kfold=KFold(n_splits=2)


# In[300]:


for train_index,val_test_index in kfold.split(x):
    print(train_index,val_test_index)


# In[301]:


from sklearn.model_selection import RepeatedKFold
rkfold=RepeatedKFold(n_splits=2,n_repeats=2,random_state=0)
for train_index,val_test_index in rkfold.split(x):
    print(train_index,val_test_index)


# In[302]:


y = [0 , 1] * 10


# In[303]:


y


# In[304]:


pd.DataFrame({'x':x,'y':y})


# In[305]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)


# In[306]:


pd.Series(y_train).value_counts()


# In[307]:


kfold=KFold(n_splits=2)
for train_index,val_test_index in kfold.split(x):
    print(train_index,val_test_index)


# In[312]:


from sklearn.model_selection import StratifiedKFold
skfold=StratifiedKFold()
for train_index,val_test_index in skfold.split(x,y):
    print(train_index,val_test_index)


# In[ ]:




