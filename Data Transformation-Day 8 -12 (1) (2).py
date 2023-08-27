#!/usr/bin/env python
# coding: utf-8

# In[52]:


import pandas as pd


# In[53]:


athlete_data=pd.read_csv('athleteData_cleaned.csv')
modified_data=pd.read_csv('modifiedData_cleaned.csv')


# In[54]:


modified_data
modified_data= modified_data.drop(['Unnamed: 0'],axis=1)
modified_data


# ## Numerical data types
# 
# Discrete: To explain in simple terms, any numerical data that is countable is called discrete, for example, the number of people in a family or the number of students in a class. Discrete data can only take certain values (such as 1, 2, 3, 4, etc)
# 
# 
# Continuous: Any numerical data that is measurable is called continuous, for example, the height of a person or the time taken to reach a destination. Continuous data can take virtually any value (for example, 1.25, 3.8888, and 77.1276).

# ## Categorical Data Types
# 
# Ordered: Any categorical data that has some order associated with it is called ordered categorical data, for example, movie ratings (excellent, good, bad, worst) and feedback (happy, not bad, bad). You can think of ordered data as being something you could mark on a scale.
# 
# 
# Nominal: Any categorical data that has no order is called nominal categorical data. Examples include gender and country.

# ## Handling Categorical Data
# There are some algorithms that can work well with categorical data, such as decision trees. But most machine learning algorithms cannot operate directly with categorical data. These algorithms require the input and output both to be in numerical form. If the output to be predicted is categorical, then after prediction we convert them back to categorical data from numerical data. Let's discuss some key challenges that we face while dealing with categorical data:

# <font color= blue>High cardinality</font>: Cardinality means uniqueness in data. The data column, in this case, will have a lot of different values. A good example is User ID – in a table of 500 different users, the User ID column would have 500 unique values.
# 
# 
# <font color = blue>Rare occurrences</font>: These data columns might have variables that occur very rarely and therefore would not be significant enough to have an impact on the model.
# 
# 
# <font color= blue>Frequent occurrences</font>: There might be a category in the data columns that occurs many times with very low variance, which would fail to make an impact on the model.
# 
# 
# <font color= blue>Won't fit</font>: This categorical data, left unprocessed, won't fit our model.

# ## Encoding
# 
# To address the problems associated with categorical data, we can use encoding. This is the process by which we convert a categorical variable into a numerical form. Here, we will look at three simple methods of encoding categorical data.
# 
# <font color =blue>Replacing</font>
# 
# This is a technique in which we replace the categorical data with a number. This is a simple replacement and does not involve much logical processing. Let's look at an exercise to get a better idea of this.
# 
# Let's try replacing technique on our dataset

# In[5]:


modified_data['Sex'].value_counts()


# In[6]:


modified_data['Sex'].replace({'M':0,'F':1},inplace=True)


# In[7]:


modified_data


# In[13]:


len(set(modified_data['Medal']))


# In[14]:


modified_data['Medal'].replace({'None':0,'Bronze':1,'Silver':2,'Gold':3},inplace=True)


# In[15]:


modified_data


# <font color=blue>Label Encoding</blue>
# 
# This is a technique in which we replace each value in a categorical column with numbers from 0 to N-1. For example, say we've got a list of employee names in a column. After performing label encoding, each employee name will be assigned a numeric label. But this might not be suitable for all cases because the model might consider numeric values to be weights assigned to the data. Label encoding is the best method to use for ordinal data. The scikit-learn library provides LabelEncoder(), which helps with label encoding. 
# 
# Converting Categorical Data to Numerical Data Using Label Encoding in our dataset

# In[16]:


import numpy as np
data_column_category= modified_data.select_dtypes(exclude=[np.number]).columns


# In[17]:


from sklearn.preprocessing import LabelEncoder

#Creating the object instance

label_encoder = LabelEncoder()

for i in data_column_category:

    modified_data[i] = label_encoder.fit_transform(modified_data[i])

print("Label Encoded Data: ")

modified_data.head()


# In[19]:


modified_data.columns


# ## Now that we have all numerical variables, let's deep dive into normalization

# Data Normalization is a common practice in machine learning which consists of transforming numeric columns to a common scale. In machine learning, some feature values differ from others multiple times. The features with higher values will dominate the leaning process. However, it does not mean those variables are more important to predict the outcome of the model. Data normalization transforms multiscaled data to the same scale. After normalization, all variables have a similar influence on the model, improving the stability and performance of the learning algorithm

# There are multiple normalization techniques in statistics: 
# 
# 
# The min-max feature scaling
# 
# 
# Standardization

# Min-Max Normalization 
# Here, all the values are scaled in between the range of [0,1] where 0 is the minimum value and 1 is the maximum value. The formula for Min-Max Normalization is –
# 
# ![image.png](attachment:image.png)

# In[21]:


set(df_scaled['Medal'])


# ## Group By & Statistical Functions

# In[25]:


modified_data['Age'].mean()


# In[26]:


modified_data[['Age','Height']].median()


# In[27]:


modified_data[['Age','Height']].describe()


# In[31]:


modified_data[['Age','Height','Sex']].groupby('Sex').mean()


# In[32]:


modified_data.groupby('Sex').mean()


# In[34]:


modified_data[modified_data['Sex']=='F'].mean()['Age']


# ## Merge function in Pandas

# Read the official doc from here:
# 
# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.merge.html

# Format:
# 
# DataFrame.merge(right, how='inner', on=None, left_on=None, right_on=None, left_index=False, right_index=False, sort=False, suffixes=('_x', '_y'), copy=True, indicator=False, validate=None)

# In[22]:


import pandas as pd
df1 = pd.DataFrame({'key': ['foo', 'bar', 'baz', 'abc'],
                    'value': [1, 2, 3, 5]})
df2 = pd.DataFrame({'key': ['foo', 'efg', 'fgh', 'ghi'],
                    'value': [1, 2, 3, 5]})


# In[34]:


df1


# In[24]:


df2


# In[26]:


df1.merge(df2, how="left", on='key')


# ## Join function in Pandas

# Format:
# 
# DataFrame.join(other, on=None, how='left', lsuffix='', rsuffix='', sort=False)

# In[41]:


df = pd.DataFrame({'key':['K0','K1','K2','K3','K4','K5'],
                  'A':['A0','A1','A2','A3','A4','A5']})


# In[42]:


df


# In[43]:


other = pd.DataFrame({'key':['K0','K1','K2'],
                  'A':['B0','B1','B2']})


# In[44]:


other


# ### joining by indexes

# In[32]:


df.join(other, on=None, how='left', lsuffix='_caller', rsuffix='_other', sort=True)


# In[33]:


df.join(other.set_index('key'), on='key')


# ## That's okay but what is the difference between merge & join then?

# Essentially both serve the same purpose- joining two tables. However,merge is more versatile, it requires specifying the columns as a merge key. We can specify the overlapping columns with parameter on, or can separately specify it with left_on and right_on parameters.

# ## Concat Function in Pandas

# Documentation:
# 
# https://pandas.pydata.org/docs/reference/api/pandas.concat.html

# pandas.concat(objs, axis=0, join='outer', ignore_index=False, keys=None, levels=None, names=None, verify_integrity=False, sort=False, copy=True)

# In[70]:


pd.concat([df,other],axis=0)


# In[56]:


head=modified_data.head()
tail=modified_data.tail()


# In[57]:


pd.concat([head,tail],axis=0)


# ## Quick practice!

# Q1) Create your own dataframe with 3 columns-- marks in Math, Science & English with 3 rows with student names

# In[68]:


data=pd.DataFrame(data={'Math':[78,85,96],'Science':[84,94,89],'English':[86,97,96]}, index=["Arpan","Suraj","Sheetal"])


# In[69]:


data


# Q2)Write a Pandas program to join the two given dataframes along rows and assign all data.
# 

# In[74]:


student_data1 = pd.DataFrame({
        'student_id': ['S1', 'S2', 'S3', 'S4', 'S5'],
         'name': ['Danniella Fenton', 'Ryder Storey', 'Bryce Jensen', 'Ed Bernal', 'Kwame Morin'], 
        'marks': [200, 210, 190, 222, 199]})

student_data2 = pd.DataFrame({
        'student_id': ['S4', 'S5', 'S6', 'S7', 'S8'],
        'name': ['Scarlette Fisher', 'Carla Williamson', 'Dante Morse', 'Kaiser William', 'Madeeha Preston'], 
        'marks': [201, 200, 198, 219, 201]})

x=pd.concat([student_data1,student_data2])
x.reset_index(inplace=True, drop=True)
x


# Q3) Write a Pandas program to join the two dataframes with matching records from both sides where available.

# In[75]:


merged_data=pd.merge(student_data1, student_data2, on='student_id', how='outer')
merged_data


# Q4)Write a Pandas program to detect missing values of a given DataFrame. Display True or False. Go to the editor
# 

# In[76]:


merged_data.isna()


# Q5)Write a Pandas program to split the following dataframe by student id and get mean, min, and max value of marks for each student
# 

# In[86]:


x.groupby('student_id').mean()['marks']
x.groupby('student_id').min()['marks']
x.groupby('student_id').max()['marks']


grp= x.groupby('student_id').agg({'marks':['mean','min','max','median']})
grp


# Q6)Write a Pandas program to drop the rows where at least one element is missing in a given DataFrame

# In[87]:


merged_data.dropna()


# ## from abc import *

# In[88]:


from numpy import *
a= arange(15)
a


# In[89]:


b=array([1,5,3,2])
b


# In[90]:


import numpy as np
a=np.arange(15)
a


# In[91]:


b=np.array([1,5,3,2])
b


# ## Sorting & Indexing Dataframe

# In[94]:


modified_data.sort_values(by=['City'], ascending=False)


# In[95]:


modified_data.sort_values(by=['Weight'], ascending=False)


# In[96]:


athlete= pd.read_csv('athlete_events.csv')


# In[97]:


athlete.sort_values(by=['Medal'], na_position='first')


# In[98]:


modified_data.sort_values(by=['Height','Weight'], ascending=False)


# ## Question: Return Top 3 heaviest weight player rows

# In[99]:


modified_data.sort_values(by=['Weight'], ascending=False).head(3)


# ## Official Documentation:: 
# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.sort_values.html

# In[2]:


import pandas as pd
df = pd.DataFrame({
   "time": ['0hr', '128hr', '72hr', '48hr', '96hr'],
   "value": [10, 20, 30, 40, 50]
})
df


# In[3]:


df.sort_values(by='time')


# In[6]:


import numpy as np
from natsort import index_natsorted
df.sort_values(
   by="time",
   key=lambda x: np.argsort(index_natsorted(df["time"]))
)


# In[5]:


get_ipython().system(' pip install natsort')


# ## Indexing in Pandas

# Pandas Indexing using [ ], .loc[], .iloc[ ], .ix[ ]
# There are a lot of ways to pull the elements, rows, and columns from a DataFrame. There are some indexing method in Pandas which help in getting an element from a DataFrame. These indexing methods appear very similar but behave very differently. Pandas support four types of Multi-axes indexing they are:
# 
# Dataframe.[ ] ; This function also known as indexing operator
# 
# 
# Dataframe.loc[ ] : This function is used for labels.
# 
# 
# Dataframe.iloc[ ] : This function is used for positions or integer based
# 
# 
# Dataframe.ix[] : This function is used for both label and integer based
# 
# 
# Collectively, they are called the indexers. These are by far the most common ways to index data. These are four function which help in getting the elements, rows, and columns from a DataFrame.

# ## <font color =blue>Indexing using []

# In[8]:


modifiedData= pd.read_csv('modifiedData_cleaned.csv')
age= modifiedData['Age']
age


# ### Selecting multiple columns

# In[9]:


num_variables= modifiedData[['Age','Height','Weight']]
num_variables


# ## <font color =blue>Indexing using df.loc

# Indexing a DataFrame using .loc[ ] :
# This function selects data by the label of the rows and columns. The df.loc indexer selects data in a different way than just the indexing operator. It can select subsets of rows or columns. It can also simultaneously select subsets of rows and columns.

# ### Selecting a Single Row

# In[66]:


df = pd.DataFrame([[1, 2], [4, 5], [7, 8]],
     index=['cobra', 'viper', 'sidewinder'],
     columns=['max_speed', 'shield'])
df


# In[11]:


df.loc['viper']


# In[14]:


locEx= modifiedData.loc[2]
locEx


# In[27]:


df.loc[['viper','sidewinder']]


# In[17]:


df.loc['viper','max_speed']


# In[31]:


df.loc['viper':'sidewinder']


# In[24]:


df.loc[[False, False, True]]


# In[36]:


df.loc[pd.Index(['cobra','viper'], name='Cobra_Viper')]


# In[37]:


df


# In[64]:


df.loc[df['shield']>6, ['shield']]


# In[38]:


# setting values
df.loc['cobra','shield']=10


# In[39]:


df


# In[52]:


df.loc['max_speed']= 15


# In[50]:


df.loc['cobra',:]= 5


# In[67]:


df.loc[df['shield']>6, ['shield']] = 0
df


# ## df.iloc

# .iloc[] is primarily integer position based (from 0 to length-1 of the axis), but may also be used with a boolean array

# In[102]:


df = pd.DataFrame([[1, 'test'], [4, 'hi'], [7, 'text']],
     index=['cobra', 'viper', 'sidewinder'],
     columns=['max_speed', 'shield'])
df


# In[76]:


df.iloc[1:2,:]


# In[77]:


df.loc['viper':,:]


# In[84]:


df.iloc[[0]]


# In[83]:


df.iloc[0:1]


# In[85]:


df.iloc[[0,1]]


# In[ ]:


df.loc[['cobra','viper']]


# In[86]:


df.iloc[:3]


# In[104]:


df[df.shield.str.startswith('t')]


# In[105]:


modifiedData[modifiedData.City.str.startswith('A')]


# In[106]:


names=['Greece','United States','France']
modifiedData[modifiedData['Team'].isin(names)]


# In[107]:


modifiedData[modifiedData.City.str.contains('y')]


# In[112]:


#Tilde

modifiedData[~modifiedData.City.str.startswith('A')]


# In[119]:


modifiedData.query('Height > = 180 and Team == "India"')


# In[120]:


modifiedData.nsmallest(2, 'Height')


# In[122]:


modifiedData.nlargest(5, 'Weight')


# ## Pivot Table

# pd.pivot_table(dataframe, values='the column with which you want to populate dataframe', index= columns grouped by which you want to aggregate,columns=column name with all categorical variables, aggfunc=the aggregation(mean/median/sum/min/max)
# fill_values= the value you want to fill NaN values with)

# In[ ]:





# In[3]:


# Using sum in pivot table


# In[ ]:





# In[ ]:





# In[ ]:





# In[4]:


## practice!


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## Changing Datatypes in Pandas

# You have four main options for converting types in pandas:
# 
# to_numeric() - provides functionality to safely convert non-numeric types (e.g. strings) to a suitable numeric type. (See also to_datetime() and to_timedelta().)
# 
# astype() - convert (almost) any type to (almost) any other type (even if it's not necessarily sensible to do so). Also allows you to convert to categorial types (very useful).
# 
# infer_objects() - a utility method to convert object columns holding Python objects to a pandas type if possible.
# 
# convert_dtypes() - convert DataFrame columns to the "best possible" dtype that supports pd.NA (pandas' object to indicate a missing value).

# ### to_numeric()

# In[130]:


s= pd.Series(['1.5','2',-3])
s.dtypes
pd.to_numeric(s)


# In[131]:


pd.to_numeric(s, downcast='integer')


# In[128]:


s= pd.Series(['apple','1.0','2',-3])
pd.to_numeric(s, errors='ignore')


# In[129]:


pd.to_numeric(s, errors='coerce')


# ### astype()

# In[133]:


modifiedData['Height'] = modifiedData['Height'].astype(int)


# In[135]:


modifiedData['Height'].dtypes


# ### infer_objects()

# Tries to get better data types for the OBJECT datatypes

# In[136]:


modifiedData.dtypes


# In[137]:


modifiedData.infer_objects().dtypes


# In[141]:


df= pd.DataFrame({'A':["a",1,2,3]})
df= df.iloc[1:]
df.dtypes


# In[142]:


df.infer_objects().dtypes


# ### convert_dtypes()

# In[143]:


df = pd.DataFrame(
    {
        "a": pd.Series([1, 2, 3], dtype=np.dtype("int32")),
        "b": pd.Series(["x", "y", "z"], dtype=np.dtype("O")),
        "c": pd.Series([True, False, np.nan], dtype=np.dtype("O")),
        "d": pd.Series(["h", "i", np.nan], dtype=np.dtype("O")),
        "e": pd.Series([10, np.nan, 20], dtype=np.dtype("float")),
        "f": pd.Series([np.nan, 100.5, 200], dtype=np.dtype("float")),
    }
)


# In[144]:


df


# In[145]:


df.dtypes


# In[146]:


df_new= df.convert_dtypes()


# In[147]:


df_new


# In[148]:


df_new.dtypes

