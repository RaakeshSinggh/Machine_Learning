#!/usr/bin/env python
# coding: utf-8

# In[35]:


##import data
import pandas as pd
data=pd.read_csv('heart.csv')
data

# downlosd from https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset


# In[36]:


# display head, tail, describe, info,rows, columns, no. of unique elemnets in each column etc
data.head()
data.tail()
data.describe()
data.info
data.columns
len(data)
#list(set(list(data.coulumn_name)))


# In[37]:


## check duplicates and drop them +reset index
data=data.drop_duplicates()
data.reset_index(inplace=True, drop=True)
data


# In[38]:


## check NaN values in each column and filter rows based on them
data.isna().sum()


# In[39]:


# check each datatype & find if it makes sense. Use astype/to_numeric to convert
data.dtypes


# In[40]:


## search for outliers using box plots
import seaborn as sns
sns.boxplot(data['thal'])


# In[41]:


# remove above 180--trestbps, --500:chol less than 80 --thalach, >5--oldpeak >27 --ca, <0.5--thal

#data[data['trestbps']>180]
#data=data[data['ca']>0.5]
data


# In[42]:


# try finding correlation between columns- and eliminate if representing same info
data.corr()


# In[33]:


# check out meaning of each column in Kaggle


# In[7]:


# find mean, 25%, 75%, median, mode etc all info from all columns


# In[43]:


# assume target is the target variable here-- find out the top 3 highest correlated features & only keep them
new_df= data[['cp','restecg','slope','target']]


# In[44]:


# find out average age of people having a cp >1
data[data['cp']>1]['age'].mean()


# In[45]:


## try doing seaborn-- all plots
import seaborn as sns
## plot a catplot between age/cholestrol -- hue= cp col=sex- kind='bar'-- if you get an inconclusive data- try grouping rows
##-- and creating a separate column called age range

g=sns.catplot(x="age", y="chol", hue="target", col="sex", data=data)


# In[46]:


# All plots learned till yesterday
sns.pairplot(data)


# In[47]:


lst=[]
for age in data['age']:
    if age <=40:
        lst.append("Upto 40")
    elif age>=41 and age<=50:
        lst.append("41-50")
    elif age>=51 and age<=60:
        lst.append("51-60")
    elif age>=61 and age<=70:
        lst.append("61-70")
    else:
        lst.append("Above 71")
data['Age Range']= lst


# In[51]:


## which age range has highest fbs & cholestrol

sns.barplot(x="Age Range", y="fbs", data=data)


# In[48]:


data


# In[46]:


## try plots again on reduced features


# In[47]:


## sort values by increasing thalach


# In[ ]:


## be creative! Any specific analysis you want to come up with?


# In[10]:


# try converting categorical to numerical/ scaling data if required


# In[52]:


s= data.groupby(['sex','Age Range'])
s.mean()


# In[ ]:




