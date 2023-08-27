#!/usr/bin/env python
# coding: utf-8

# ## Correlation

# Correlation is a statistical measure that determines the association or co-relationship between two variables

# Pandas dataframe.corr() is used to find the pairwise correlation of all columns in the dataframe. Any na values are automatically excluded. For any non-numeric data type columns in the dataframe it is ignored.

# The correlation of a variable with itself is 1.

# In[1]:


import pandas as pd
url='https://media.geeksforgeeks.org/wp-content/uploads/nba.csv'
data= pd.read_csv(url)


# In[2]:


data


# In[3]:


data=data.dropna()
data.reset_index(drop=True, inplace=True)
data


# In[4]:


data.corr()


# In[3]:


import seaborn as sns
sns.regplot(x=data['Number'], y=data['Number'])


# ## Regression 

# Regression describes how to numerically relate an independent variable to the dependent variable

# Linear regression is used to predict the continuous dependent variable using a given set of independent variables. 
# 
# Logistic Regression is used to predict the categorical dependent variable using a given set of independent variables.

# ### <font color= red>You'll learn these in more detail in ML module</font>

# ## Visualization using Seaborn

# ## Why do we need visualization?

# Data visualization helps to tell stories by curating data into a form easier to understand, highlighting the trends and outliers. A good visualization tells a story, removing the noise from data and highlighting the useful information.

# ### <font color=blue>Data Visualization using External Softwares:</font>
# 
# 1) Tableau
# 
# 2) Power BI
# 
# 3) Qlikview

# ### <font color=blue>Data Visualization using Python:</font>
# 
# 1) Matplotlib
# 
# 2) Seaborn

# Let's see few differences between them:
# ![image.png](attachment:image.png)
# 
# ![image-2.png](attachment:image-2.png)

# Seaborn is a library for making statistical graphics in Python. It builds on top of matplotlib and integrates closely with pandas data structures.
# 
# Seaborn helps you explore and understand your data. Its plotting functions operate on dataframes and arrays containing whole datasets and internally perform the necessary semantic mapping and statistical aggregation to produce informative plots. Its dataset-oriented, declarative API lets you focus on what the different elements of your plots mean, rather than on the details of how to draw them.

# In[7]:


import seaborn as sns

# Load an example dataset
tips= sns.load_dataset('tips')
tips
# Create a visualization
sns.relplot(data=tips, x='total_bill',y='tip', col='time', hue='day', size='size')


# In[8]:


tips


# Relational plots are used for visualizing the statistical relationship between the data points

# ### Cat plots

# Category Plots in Seaborn
# Read the documentation:
# 
# https://seaborn.pydata.org/generated/seaborn.catplot.html

# seaborn.catplot(*, x=None, y=None, hue=None, data=None, row=None, col=None, col_wrap=None, estimator=<function mean at 0x7ff320f315e0>, ci=95, n_boot=1000, units=None, seed=None, order=None, hue_order=None, row_order=None, col_order=None, kind='strip', height=5, aspect=1, orient=None, color=None, palette=None, legend=True, legend_out=True, sharex=True, sharey=True, margin_titles=False, facet_kws=None, **kwargs)

# In[34]:


# ci= confidence interval

g= sns.catplot(data=tips, x="size", y="total_bill", kind='bar' , ci=None)
g


# In[ ]:


tips_1= [['total']]


# ### Dist plots

# A distplot plots a univariate distribution of observations. The distplot() function combines the matplotlib hist function with the seaborn kdeplot() and rugplot() functions.
# 
# Read the documentation:
# 
# https://seaborn.pydata.org/generated/seaborn.distplot.html

# In[5]:


import seaborn as sns, numpy as np
sns.set_theme()
x= np.random.randn(100)
sns.distplot(x, hist=False)


# In[11]:


sns.distplot(x= tips['size'])


# ### Pair Plot
# 

# Pair plot is used to understand the best set of features to explain a relationship between two variables or to form the most separated clusters. It also helps to form some simple classification models by drawing some simple lines or make linear separation in our data-set.
# 
# Read the documentation:
# 
# https://seaborn.pydata.org/generated/seaborn.pairplot.html

# In[19]:


sns.pairplot(tips)


# In[21]:


tips[['size','total_bill']]


# In[15]:


penguins = sns.load_dataset("penguins")
sns.pairplot(penguins)


# In[16]:


penguins


# In[17]:


penguins.corr()


# ### Joint Plots

# Jointplot is seaborn library specific and can be used to quickly visualize and analyze the relationship between two variables and describe their individual distributions on the same plot.
# 
# Read documentation:
# 
# https://seaborn.pydata.org/generated/seaborn.jointplot.html

# In[36]:


sns.jointplot(data=tips, x="total_bill", y='tip', hue= 'sex')


# ## Bar Plot

# A bar chart or bar graph is a chart or graph that presents categorical data with rectangular bars with heights or lengths proportional to the values that they represent. The bars can be plotted vertically or horizontally.
# 
# A bar graph shows comparisons among discrete categories. One axis of the chart shows the specific categories being compared, and the other axis represents a measured value.
# 
# Documentation:
# https://seaborn.pydata.org/generated/seaborn.barplot.html

# In[52]:


import seaborn as sns
sns.barplot(x='time', y='total_bill', data=tips, hue='sex', ci=None)


# In[39]:


data= tips[['time','total_bill']]


# In[43]:


data.groupby('time').mean()


# ### Density Plot
# 

# A density plot is a representation of the distribution of a numeric variable. It uses a kernel density estimate to show the probability density function of the variable (see more). It is a smoothed version of the histogram and is used in the same concept.
# 
# ![image.png](attachment:image.png)
# 
# Read documentation here:
# 
# https://seaborn.pydata.org/generated/seaborn.kdeplot.html

# In[54]:


sns.kdeplot(data=tips,x='total_bill',y='tip')+

#rug plot


# ## This was an overview:: Please try doing all the plots in all the available datasets on Seaborn
