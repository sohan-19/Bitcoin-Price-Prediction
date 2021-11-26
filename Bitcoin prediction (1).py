#!/usr/bin/env python
# coding: utf-8

# ### <center> Imported The Libraries

# In[1]:


import pandas as pd
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt
import statsmodels.api as sm


# ### <center> Loaded The Dataset

# In[2]:


Bitcoin = pd.read_csv("C://Users//91900//Desktop//Data//bitcoin_dataset.csv")


# In[3]:


Bitcoin


# In[4]:


# finding the value of 1024th in btc_market_price column
Bitcoin.loc[1024]


# ### <center> EDA

# In[5]:


Bitcoin.shape


# In[6]:


Bitcoin.isnull().sum()


# In[7]:


Bitcoin.tail()


# In[8]:


from pandas_profiling import ProfileReport
Bitcoin.profile_report()


# In[9]:


corr=Bitcoin.corr( )
corr


# In[10]:


bitcoin=sns.heatmap(corr,annot=True)


# In[11]:


# In below plots we can see the coreelation value  in top right of the graph.
# In this graph btc_market_price is correlated with btc_market_cap
sns.jointplot(data=Bitcoin,x="btc_market_price", y="btc_market_cap")


# In[12]:


sns.jointplot(data=Bitcoin, x='btc_market_price', y="btc_hash_rate")


# ### <center> Preprocessing

# In[13]:


Bitcoin['Date']=pd.to_datetime(Bitcoin['Date'],format='%m/%d/%Y')
Bitcoin['year']=Bitcoin['Date'].dt.year
Bitcoin['month']=Bitcoin['Date'].dt.month
Bitcoin['day']=Bitcoin['Date'].dt.day


# In[14]:


Bitcoin1=Bitcoin.head()
Bitcoin1


# In[15]:


Bitcoin1=Bitcoin.drop(['Date'],axis=1)
Bitcoin1


# In[16]:


Bitcoin1.info()


# In[17]:


Bitcoin1.median()


# In[18]:


Bitcoin1['btc_trade_volume']=Bitcoin1['btc_trade_volume'].fillna(Bitcoin1['btc_trade_volume'].median())
Bitcoin1['btc_total_bitcoins']=Bitcoin1['btc_total_bitcoins'].fillna(Bitcoin1['btc_total_bitcoins'].median())
Bitcoin1['btc_blocks_size']=Bitcoin1['btc_blocks_size'].fillna(Bitcoin1['btc_blocks_size'].median())
Bitcoin1['btc_median_confirmation_time']=Bitcoin1['btc_median_confirmation_time'].fillna(Bitcoin1['btc_median_confirmation_time'].median())
Bitcoin1['btc_difficulty']=Bitcoin1['btc_difficulty'].fillna(Bitcoin1['btc_difficulty'].median())
Bitcoin1['btc_transaction_fees']=Bitcoin1['btc_transaction_fees'].fillna(Bitcoin1['btc_transaction_fees'].median())


# In[19]:


Bitcoin1.isnull().sum()


# ### <center> Model Formulation & Predictions

# In[20]:


y=Bitcoin1['btc_market_price']
x=Bitcoin1.drop(['btc_market_price'],axis=1)


# In[21]:


y


# In[22]:


x


# In[23]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=240)


# In[24]:


x_train


# In[25]:


y_train


# In[26]:


from sklearn.linear_model import LinearRegression
reg=LinearRegression()
model=reg.fit(x_train,y_train)
y_pred=model.predict(x_test)


# In[27]:


from sklearn import metrics
print('Mean squared error',metrics.mean_squared_error(y_test,y_pred))


# In[28]:


print('coefficients: \n',model.coef_)


# In[29]:


#So if it is 100%, the two variables are perfectly correlated, i.e., with no variance at allss
print('variance score: {}',format(model.score(x_test,y_test)))

