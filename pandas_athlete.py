#!/usr/bin/env python
# coding: utf-8

# In[15]:


import pandas as pd
athletedata = pd.read_csv('athlete_events.csv')
athletedata


# In[16]:


# find the top 5 rows of your data
import pandas as pd
athletedata = pd.read_csv('athlete_events.csv')
athletedata.head()


# In[17]:


# find the last 5 rows of your data
import pandas as pd
athletedata = pd.read_csv('athlete_events.csv')
athletedata.tail()


# In[18]:


# Drop duplicates
import pandas as pd
athletedata = pd.read_csv('athlete_events.csv')
athletedata = athletedata.drop_duplicates()
athletedata = athletedata.reset_index(drop = True,inplace = False)
athletedata


# In[19]:


# checking for  null values.
import pandas as pd
athletedata = pd.read_csv('athlete_events.csv')
athletedata = athletedata.isna().sum()
athletedata


# In[20]:


import pandas as pd
athletedata = pd.read_csv('athlete_events.csv')
athletedata = set(athletedata['Medal'])
athletedata


# In[21]:


#check column names
import pandas as pd
athletedata = pd.read_csv('athlete_events.csv')
athletedata = athletedata.columns
athletedata


# In[22]:


# Viewing each column
import pandas as pd
athletedata = pd.read_csv('athlete_events.csv')
athletedata =set(list(athletedata['Age']))
athletedata


# In[23]:


#check data types
import pandas as pd
athletedata = pd.read_csv('athlete_events.csv')
athletedata = athletedata.dtypes
athletedata


# In[31]:


# #convert datatypes if required
# import pandas as pd
# import seaborn as sns
# from matplotlib import pyplot as plt
# athletedata = pd.read_csv('athlete_events.csv')
# athletedata['Age'].fillna(0)
# athletedata['Age'] = pd.to_numeric(athletedata['Age'], errors = 'coerce',).astype('int64') 
# print(athletedata.dtypes)


# # Data cleaning and Missing value Handling

# In[26]:


import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
athletedata = pd.read_csv('athlete_events.csv')
athletedata.head()
athletedata['Year'].fillna(0) # (What I found on the Internet may be that there is an empty space in the data, and NA is filled with o, but it is still not easy to use.)
athletedata['Year']= pd.to_numeric(athletedata['Year'], errors='coerce').astype('int64')
# athletedata['Height '] = athletedata['Height'].astype(int) # (How to change the data type of a column in a dataframe)
athletedata.dtypes
# ax = sns.barplot(data=df, x="Year", y="Total Goals")
# plt.show()


# In[60]:


# If the number of entries is N, you have less than 5-10% missing values-- we can eliminate those rows
import pandas as pd
athletedata = pd.read_csv('athlete_events.csv')
athletedata = athletedata[athletedata['Age'].notna()]
athletedata = athletedata.reset_index(drop = True,inplace = False)
# If the number of entries is N, you have 30-50% missing values-- we can replace the rows with mean, median, mode
athletedata['Height'] = athletedata['Height'] .fillna(athletedata['Height'].mean())
athletedata['Weight'] = athletedata['Weight'] .fillna(athletedata['Weight'].mean())
athletedata['Medal'] = athletedata['Medal'] .fillna('None')
athletedata = athletedata.reset_index(drop = True,inplace = False)
#If the number of entries is N, you have above 70% missing values-- Use some intuition and eliminate the column


athletedata


# In[62]:


# #convert if required
# import pandas as pd
# athletedata = pd.read_csv('athlete_events.csv')
# athletedata['Age']= athletedata['Age'].astype('int64')
athletedata['Age'] = athletedata['Age'].fillna(0)
athletedata['Age'] = pd.to_numeric(athletedata['Age'], errors='coerce').astype('int64')
athletedata['Height'] = athletedata['Height'].fillna(0)
athletedata['Height'] = pd.to_numeric(athletedata['Height'], errors='coerce').astype('int64')
athletedata['Weight'] = athletedata['Weight'].fillna(0)
athletedata['Weight'] = pd.to_numeric(athletedata['Weight'], errors='coerce').astype('int64')
athletedata.dtypes


# In[63]:


# import pandas as pd
import seaborn as sns
athletedata = pd.read_csv('athlete_events.csv')
athletedata = sns.boxplot(athletedata['Year'])
athletedata


# In[5]:


import pandas as pd
athletedata = pd.read_csv('athlete_events.csv')
athletedata = athletedata[athletedata['Year'] <= 1900]
athletedata


# In[6]:


# drop columns
import pandas as pd
athletedata = pd.read_csv('athlete_events.csv')
athletedata1 = athletedata.drop(['Year'],axis =1)
athletedata2 = athletedata1.drop(['Season'],axis =1)
athletedata3 = athletedata2.drop(['Sport'],axis =1)
athletedata4 = athletedata3.drop(['NOC'],axis =1)
athletedata4

# # drop duplicates
# Modified_data = athletedata.drop_duplicates()
# Modified_data.reset_index(inplace = True , drop = True)
# Modified_data


# # drop duplicates
# athletedata = athletedata.drop_duplicates()
# athletedata = athletedata.reset_index(drop = True,inplace = False)
# print(athletedata)


# In[7]:


import pandas as pd
athletedata = pd.read_csv('athlete_events.csv')
athletedata = athletedata4
athletedata4


# In[13]:


# remove name
Modified_data = athletedata4.drop(['Name'],axis = 1)
Modified_data


# In[26]:


import pandas as pd
modified_data = Modified_data.drop_duplicates()
modified_data
modified_data.to_csv('ModifiedData_cleaned.csv')
modified_data


# In[45]:


# group by
group = Modified_data.groupby('ID')
group.first()

event = Modified_data.groupby('Event')
event.mean()


#  convert categorical to numerical values
# df1 = pd.get_dummies(df['parchased'])

# In[46]:


df1 = pd.get_dummies(Modified_data['Team'])
df1


# # MODIFIED DATA

# In[51]:


# import pandas as pd
# athletedata = pd.read_csv('athlete_events.csv')
# athletedata = athletedata.drop_duplicates()
# print(athletedata)
# athletedata = athletedata.to_csv('modifiedData_cleaned.csv')
# print(athletedata)
# athletedata = athletedata.to_csv('athleteData_cleaned.csv')
# print(athletedata)


# In[73]:


# #group by
# import pandas as pd
# athletedata = pd.read_csv('athlete_events.csv')
# athletedata = athletedata.groupby('ID')
# athletedata = athletedata.first()
# athletedata

# athletedata = athletedata.groupby('Event')
# athletedata.mean()


# # Min-Max Scaler and Standard Scaler

# In[14]:


import pandas as pd
from sklearn.preprocessing import MinMaxScaler
athletedata = pd.read_csv('athlete_events.csv')
scaler= MinMaxScaler()
dftest= pd.DataFrame({'a':[1,2,3,4,5],
                     'b':[100,500,2500,12500,62500]})
dftest
dftest[['a','b']]=scaler.fit_transform(dftest[['a','b']])
dftest


# In[53]:


# Q.1 In Wrestling Women's Flyweight, Freestyle - what's the mean age of participants?

Modified_data[Modified_data['Event']=="Wrestling Women's Flyweight, Freestyle"].mean()



# In[65]:


# Q.2 Which of these are categorical & numerical variables
modified_data.dtypes


# In[85]:


# Q.3 Which participant won the maximum number of medals?

Medal = athletedata[athletedata['Medal'] != None]
Medal['Name'].mode()[0]



# In[117]:


# Q.4  Which team had maximum Gold medals in 1992 Summer Olympics?

gold_Medal = athletedata4[(athletedata4["Medal"] == "Gold") & (athletedata4["Games"] == "1992 summer")]
gold_Medal['Team'].mode()



# In[14]:


# len(set(modified_data['Medal']))


# In[108]:


modified_data['Medal'].replace({'NaN':0,'Bronze':1,'Silver':2,'Gold':3},inplace=True)
modified_data


#  Encoding

# In[8]:


import pandas as pd
athletedata4 = athletedata4['Sex'].value_counts()
athletedata4



# In[18]:


#athletedata4['Sex'].replace({'M':0,'F':1},inplace = True)


# In[28]:


# len(set(athletedata4['Team']))


#  Merge Function in Pandas

# In[32]:


import pandas as pd
df1 = pd.DataFrame({'lkey':['rakesh', 'nirdes', 'aman'],'value':[1,2,3] })
df2 = pd.DataFrame({'rkey':['ankur', 'pandey', 'jatin'],'value':[3,4,5] })
df1


# In[33]:


df2


# In[45]:


df1.merge(df2,how="left",left_on='lkey',right_on = 'rkey')


#  Join function in  Pandas

# In[47]:


df = pd.DataFrame({'key':['k0','k1','k2','k3','k4','k5'],
                  'A':['a0','a1','a2','a3','a4','a5']})
df


# In[50]:


other = pd.DataFrame({'key':['k0','k1','k2'],
                  'B':['b0','b1','b2']})
other                       


# Joining by Indexing

# In[53]:


df.join(other, on=None, how='left', lsuffix='collar', rsuffix='_other', sort=False)


# In[54]:


# concatenate function in panads
pd.concat([df,other],axis =1)


# In[58]:


data= {'Maths':[75,70,80],'Science':[90,60,80],'English':[60,70,80]}
df = pd.DataFrame(data,index=['Rakesh','Nirdes','Aman'])
df


# In[21]:


student_data1 = pd.DataFrame({'student_id':[1,2,3,4],
                             'name':['rakesh','aman','nirdesh','ankur'],
                             'marks':[88,70,60,65]})
student_data2 = pd.DataFrame({'student_id':[5,6,7,8],
                             'name':['jatin','pandey','gaurav','tanni'],
                             'marks':[70,60,75,80]})
x=pd.concat([student_data1,student_data2])
x.reset_index(inplace=True)
x


# In[22]:


# merge function using pandas
merged_data = pd.merge(student_data1, student_data2, on='student_id' , how='outer' )
merged_data


# In[26]:


merged_data.isna()
# merged_data.isna().sum()


# In[29]:


x.groupby('student_id').mean()['marks']
x.groupby('student_id').min()['marks']
x.groupby('student_id').max()['marks']


grp = x.groupby('student_id').agg({'marks':['mean','min','max']})
grp


# In[30]:


merged_data.dropna()


# In[32]:


from numpy import *
a = arange(15)
a


# In[22]:


import pandas as pd
modified_data = pd.read_csv('modifiedData_cleaned.csv')
modified_data


# In[21]:


modified_data[['Age','Height','Sex']].groupby('Sex').mean()


# In[23]:


modified_data.groupby('Sex').mean()


# In[24]:


modified_data[modified_data['Sex']=='F'].mean()['Age']


# In[25]:


modified_data[['Age','Height']].describe()


# In[26]:


modified_data[['Age','Height','Sex']].groupby('Sex').mean()


# In[27]:


modified_data[['Age','Height']].median()


# In[28]:


modified_data['Age'].mean()


# In[29]:


modified_data.sort_values(by=['City'], ascending=False)


# In[31]:


modified_data.sort_values(by=['Medal'], na_position='first')


# In[32]:


modified_data.sort_values(by=['Weight'], ascending=False).head(3)


# In[33]:


modified_data[modified_data.City.str.startswith('A')]


# In[35]:


names=['Greece','United States','France']
modified_data[modified_data['Team'].isin(names)]


# In[36]:


modified_data[modified_data.City.str.contains('y')]


# In[37]:


#Tilde

modified_data[~modified_data.City.str.startswith('A')]


# In[38]:


modified_data.query('Height > = 180 and Team == "India"')


# In[39]:


modified_data.nsmallest(2, 'Height')


# In[40]:


modified_data.nlargest(5, 'Weight')


#  Pivot Table

# pd.pivot_table(dataframe, values='the column with which you want to populate dataframe', index= columns grouped by which you want to aggregate,columns=column name with all categorical variables, aggfunc=the aggregation(mean/median/sum/min/max) fill_values= the value you want to fill NaN values with)

# In[ ]:




