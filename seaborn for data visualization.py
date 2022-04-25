#!/usr/bin/env python
# coding: utf-8

# seaborn is a statistical plotting library
# it has beautiful default styles
# it also is degigned to work very well with pandas dataframe objects
                                       DISTRIBUTION PLOTS IN SEABORN
# In[1]:


import seaborn as sns


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


tips=sns.load_dataset('tips')


# In[4]:


tips.head()


# In[5]:


sns.distplot(tips['total_bill'],kde=False,bins=40)


# In[6]:


sns.jointplot(x='total_bill',y='tip',data=tips)#kind=reg,hex,default


# In[7]:


sns.pairplot(tips,hue='sex',palette='coolwarm')


# In[8]:


sns.rugplot(['total_bill'])


# In[9]:


#kde=kernel density estimation
sns.kdeplot(tips['total_bill'])


#                                     CATEGORICAL PLOTS IN SEABORN

# In[10]:


tips.head()


# In[11]:


import numpy as np


# In[12]:


sns.barplot(x='sex',y='total_bill',data=tips,estimator=np.std)


# In[13]:


sns.countplot(x='smoker',data=tips)


# In[14]:


sns.boxplot(x='day',y='total_bill',data=tips)


# In[15]:


sns.violinplot(x='day',y='total_bill',data=tips,hue='sex')


# In[16]:


sns.stripplot(x='day',y='total_bill',data=tips,jitter=True,hue='sex')


# In[17]:


sns.violinplot(x='day',y='total_bill',data=tips)
sns.swarmplot(x='day',y='total_bill',data=tips,color='black')


# In[18]:


sns.factorplot(x='day',y='total_bill',data=tips)


#                                                 MATRIX PLOT IN SEABORN

# In[19]:


tips.head()


# In[20]:


flights=sns.load_dataset('flights')


# In[21]:


flights.head()


# In[22]:


tc=tips.corr()


# In[23]:


sns.heatmap(tc,annot=True,cmap='coolwarm')


# In[24]:


tc


# In[25]:


fp=flights.pivot_table(index='month',columns='year',values='passengers')


# In[26]:


sns.heatmap(fp,cmap='cool',linecolor='white',linewidth=3)


# In[27]:


sns.clustermap(fp,cmap='coolwarm',linecolor='white',linewidth=3,standard_scale=1)


#                                              GRIDS IN SEABORN

# In[28]:


data=sns.load_dataset('iris')


# In[29]:


data.head()


# In[30]:


sns.pairplot(data)


# In[31]:


sns.countplot(x='species',data=data,hue='species')


# In[32]:


from matplotlib import pyplot as plt


# In[33]:


g=sns.PairGrid(data)
g.map_diag(plt.plot)
g.map_upper(sns.violinplot)
g.map_lower(plt.scatter)

                                       REGRESSION PLOT IN SEABORN 
# In[34]:


tips


# In[35]:


sns.lmplot(x='total_bill',y='tip',data=tips,hue='sex',markers=['o','v'],scatter_kws={'s':50})


# In[36]:


sns.lmplot(x='total_bill',y='tip',data=tips,col='day',row='time',hue='sex')

                                           COLOURS AND STYLES
# In[37]:


#sns.set_style('ticks')
sns.set_context('notebook')
sns.countplot(data=tips,x='sex')

#sns.despine(left=True,bottom=True)


# In[38]:


sns.lmplot(x='total_bill',y='tip',data=tips,hue='sex',palette='seismic')


# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!EXERSICE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# In[39]:


import pandas as pd


# In[40]:


titanic=sns.load_dataset('titanic')


# In[41]:


titanic.head(10)


# In[42]:


sns.jointplot(x='fare',y='age',data=titanic)


# In[43]:


sns.distplot(titanic['fare'],kde=False,color='red',bins=30)


# In[ ]:





# In[ ]:





# In[ ]:




