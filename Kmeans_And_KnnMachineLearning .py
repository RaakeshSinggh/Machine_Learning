#!/usr/bin/env python
# coding: utf-8

#  KMeans Machine Learning

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
data = load_iris()


# In[9]:


data.keys()


# In[10]:


data.feature_names


# In[11]:


data.target_names


# In[12]:


df = pd.DataFrame(data.data,columns=data.feature_names)


# In[13]:


df


# In[14]:


input_data = data['data']


# In[15]:


df = pd.DataFrame(input_data)


# In[16]:


df


# In[17]:


df.columns = data['feature_names']
df


# In[18]:


data['target']


# In[19]:


from sklearn.decomposition import PCA


# In[20]:


pca = PCA(2)


# In[21]:


df_decomposed = pd.DataFrame(pca.fit_transform(input_data))


# In[22]:


df_decomposed


# In[23]:


import matplotlib.pyplot as plt


# In[25]:


plt.scatter(df_decomposed[0],df_decomposed[1])


# In[26]:


from sklearn.cluster import KMeans
Kmeans = KMeans(n_clusters=2)


# In[27]:


Kmeans.fit(input_data)


# In[28]:


Kmeans.labels_


# In[29]:


df['cluster'] = Kmeans.labels_ 
df


# In[30]:


df.shape


# In[31]:


df.columns


# In[32]:


Kmeans.inertia_


# In[33]:


# Elbow  Methods
error = []
for k in range(1, 11):
    Kmeans = KMeans(n_clusters=k)
    Kmeans.fit(input_data)
    error.append(Kmeans.inertia_)
    
    


# In[34]:


plt.plot(error)


# In[35]:


df['target'] = data.target
df.head()


# In[36]:


df[df.target==0].head()


# In[37]:


df[df.target==1].head()


# In[38]:


df[df.target==2].head()


# In[39]:


df['flower_name'] = df.target.apply(lambda x: data.target_names[x])
df.head(150)


# In[40]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[41]:


df0=df[:50]
df1=df[50:100]
df2=df[100:]


# sepal lenght vs sepal width (setosa vs versicolor)

# In[42]:


plt.xlabel('Sepal Lenght')
plt.ylabel('Sepal Width')
plt.scatter(df0['sepal length (cm)'], df0['sepal width (cm)'], color="green",marker='+')
plt.scatter(df1['sepal length (cm)'], df1['sepal width (cm)'], color ="blue",marker='.')
plt.scatter(df2['sepal length (cm)'], df2['sepal width (cm)'], color ="red",marker='*')
plt.show()


# In[43]:


plt.xlabel('Patel Lenght')
plt.ylabel('Patel Width')
plt.scatter(df0['petal length (cm)'], df0['petal width (cm)'], color="green",marker='+')
plt.scatter(df1['petal length (cm)'], df1['petal width (cm)'], color ="blue",marker='.')
plt.scatter(df2['petal length (cm)'], df2['petal width (cm)'], color ="red",marker='*')
plt.show()


#  train test split

# In[44]:


from sklearn.model_selection import train_test_split


# In[45]:


x = df.drop(['target','flower_name'], axis='columns')
y = df.target


# In[89]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=1)


# In[48]:


len(x_train)


# In[49]:


len(x_test)


# create KNN (K Neighrest NeighborsClassifier )

# In[50]:


from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=3)


# In[51]:


knn.fit(x_train,y_train)


# In[52]:


knn.fit(x_test,y_test)


# In[53]:


knn.score(x_test,y_test)


# In[54]:


from sklearn.metrics import confusion_matrix
y_pred=knn.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
cm


# In[55]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
plt.figure(figsize=(7,5))
sns.heatmap(cm,annot=True)
plt.xlabel('Predict')
plt.ylabel('Truth')


# In[56]:


from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))


# In[63]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler


# In[65]:


url = 'https://raw.githubusercontent.com/codebasics/py/master/ML/13_kmeans/income.csv'
df = pd.read_csv(url)


# In[66]:


df.head()


# In[67]:


plt.scatter(df.Age,df['Income($)'])
plt.xlabel('Age')
plt.ylabel('Income')


# In[68]:


knn = KMeans(n_clusters=3)
y_predicted = knn.fit_predict(df[['Age','Income($)']])


# In[69]:


y_predicted


# In[70]:


df['cluster']= y_predicted
df.head()


# In[71]:


knn.cluster_centers_


# In[72]:


df1 = df[df.cluster==0]
df2 = df[df.cluster==1]
df3 = df[df.cluster==2]


# In[74]:


plt.scatter(df1.Age,df1['Income($)'],color='green',marker='+')
plt.scatter(df2.Age,df2['Income($)'],color='red',marker='.')
plt.scatter(df3.Age,df3['Income($)'],color='blue',marker='*')
plt.scatter(knn.cluster_centers_[:,0],knn.cluster_centers_[:,1],color='black',marker='*',label='centroid')
plt.xlabel('Age')
plt.ylabel('Income')


# In[75]:


scaler = MinMaxScaler()

scaler.fit(df[['Income($)']])
df['Income($)'] = scaler.transform(df[['Income($)']])

scaler.fit(df[['Age']])
df['Age'] = scaler.transform(df[['Age']])


# In[76]:


df.head()


# In[77]:


plt.scatter(df.Age,df['Income($)'])
plt.xlabel('Age')
plt.ylabel('Income')


# In[78]:


knn = KMeans(n_clusters=3)
y_predicted = knn.fit_predict(df[['Age','Income($)']])
y_predicted


# In[79]:


df['cluster']= y_predicted
df.head()


# In[80]:


knn.cluster_centers_


# In[81]:


df1 = df[df.cluster==0]
df2 = df[df.cluster==1]
df3 = df[df.cluster==2]


# In[82]:


plt.scatter(df1.Age,df1['Income($)'],color='green',marker='+')
plt.scatter(df2.Age,df2['Income($)'],color='red',marker='.')
plt.scatter(df3.Age,df3['Income($)'],color='blue',marker='*')
plt.scatter(knn.cluster_centers_[:,0],knn.cluster_centers_[:,1],color='black',marker='*',label='centroid')
plt.xlabel('Age')
plt.ylabel('Income')


# In[83]:


# Elbow  Methods
error = []
rng = range(1,10)
for k in rng:
    Knn = KMeans(n_clusters=k)
    Knn.fit(df[['Age','Income($)']])
    error.append(Knn.inertia_)


# In[92]:


error


# In[84]:


plt.xlabel('k')
plt.ylabel('sum of squared error')
plt.plot(rng,error)


# In[85]:


error


# In[86]:


from sklearn.metrics import r2_score


# In[87]:


r2_score(y_test, y_pred)


# In[93]:


# knn.score(x_test, y_test)


#  Dealing Heterogenous Data,
#  Case Studies for Bias, Variance, Validation and Hyper-parameters

#  KNN Machine Learning

# In[3]:


import pandas as pd
import numpy as np


# In[4]:


df = pd.read_csv('Social_Network_Ads.csv')


# In[5]:


df


# In[6]:


df.drop(['User ID','Purchased'],axis=1)


# In[7]:


df = pd.DataFrame({'x1': [1, 1, 1, 5, 5, 5], 'x2' : [1, 2, 3, 1, 2, 3], 'class':['c1', 'c1', 'c1', 'c2', 'c2', 'c2']})


# In[8]:


df


# In[9]:


import matplotlib.pyplot as plt


# In[10]:


plt.xlabel('x1')
plt.ylabel('x2')
plt.scatter(df['x1'], df['x2'])


# In[11]:


ip = [5, 2.25]  # [x1, y1], [x2, y2]


# In[12]:


def distance(y): # list
  return ((ip[0]-y[0])**2 + (ip[1]-y[1])**2)**(1/2)


# In[13]:


df['dist_from_ip'] = df[['x1', 'x2']].apply(distance, axis=1) # [1,	1]


# In[14]:


df


# In[15]:


k = 3


# In[23]:


df.sort_values(['dist_from_ip'])


# In[48]:


# c2 = 3, c1 = 0 for k = 3 # c2
# c2 = 3, c1 = 2 for k = 5 # c2


# In[31]:


(17.5625)**(1/2)


# In[33]:


from sklearn.neighbors import KNeighborsClassifier


# In[34]:


knn = KNeighborsClassifier()


# In[35]:


x=df[['x1','x2']]


# In[43]:


y=df['class']


# In[44]:


df['class']=[0,0,0,1,1,1]


# In[45]:


y


# In[46]:


knn.fit(x,y)


# In[47]:


knn.predict([[5,2.25]])


#  Implementing KNN using numpy from scratch

# In[49]:


import pandas as pd
import numpy as np


# In[50]:


df=pd.read_csv('Social_Network_Ads.csv')


# In[51]:


df


# In[52]:


x=df[['Age','EstimatedSalary']].values


# In[53]:


y=df['Purchased']


# In[54]:


y


# In[55]:


from sklearn.model_selection import train_test_split


# In[56]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.2,random_state=0)


# In[57]:


x_train.shape,x_test.shape


# In[58]:


x_train


# In[59]:


x_test


# In[60]:


k = 5


# In[64]:


def get_eucl_dist(l1,l2):
    return np.sqrt(np.sum((l1-l2)**2))


# In[78]:


from collections import Counter


# In[82]:


y = [0, 1, 0, 1, 0]
Counter(y).most_common(1)[0][0]


# In[83]:


def get_class(test_data):
    distances = [get_eucl_dist(train,test_data) for train in x_train]
    sort_indexes = np.argsort(distances)[:k]
    labels = [y_train.values[i] for i in sort_indexes]
    return Counter(labels).most_common(1)[0][0]
    


# In[84]:


predicted_labels = []
for test in x_test:
    predicted_labels.append(get_class(test))
    


# In[85]:


predicted_labels


# In[87]:


y_test.values


# In[88]:


np.sum(y_test.values==predicted_labels)/len(y_test.values)*100


# In[89]:


for t,i,j in zip(x_test,y_test,predicted_labels):
    print(t,'::', i, '--------------->', j)


# In[94]:


tp=0
tn=0
fp=0
fn=0
for i,j in zip(y_test,predicted_labels):
    if i==1 and j==1:
        tp+=1
    elif i==1 and j==0:
        fn+=1
    elif i==0 and j==0:
        tn+=1
    else:
        fp+=1
    print(i, j)    
            


# In[95]:


acc = (tp+tn)/(tp+tn+fn+fp)


# In[96]:


acc


# In[97]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()


# In[98]:


knn.fit(x_train,y_train)


# In[99]:


y_pred = knn.predict(x_test)


# In[100]:


y_pred


# In[101]:


len(y_pred),len(predicted_labels)


# In[103]:


for i,j in zip(y_pred,predicted_labels):
    if i!= j:
        print(i ,j)


# In[114]:


from sklearn.metrics import accuracy_score


# In[115]:


ac = accuracy_score(y_test, y_pred)


# In[116]:


ac


# In[117]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# In[118]:


cm = confusion_matrix(y_test,y_pred)


# In[119]:


cm


# In[120]:


cmd = ConfusionMatrixDisplay(cm)


# In[122]:


cmd.plot()

