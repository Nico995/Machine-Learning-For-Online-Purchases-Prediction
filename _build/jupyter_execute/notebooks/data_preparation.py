#!/usr/bin/env python
# coding: utf-8

# # Data preparation

# In[1]:


import pandas as pd
df = pd.read_csv("../dataset/online_shoppers_intention.csv")

df.head()


# In[2]:


from sklearn.preprocessing import OrdinalEncoder
textual_columns = ['Month', 'VisitorType', 'Weekend', 'Revenue']
enc = OrdinalEncoder()
df[textual_columns] = enc.fit_transform(df[textual_columns])

df.head()


# In[3]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(df.drop(columns=['Revenue']), df['Revenue'], test_size=0.2, random_state=42)

print(f'Starting size: {df.shape}   =>  Training size: {x_train.shape} , Test size: {x_test.shape}')


# In[4]:


from imblearn.over_sampling import SMOTENC

categorical_features = ['Month', 'OperatingSystems', 'Browser', 'Region', 'TrafficType', 'VisitorType', 'Weekend']
sm = SMOTENC(categorical_features=[c in categorical_features for c in df_train.columns])

x_train_resampled, y_train_resampled = sm.fit_resample(x_train, y_train)
# df_train_resampled = pd.concat((x_train_resampled, y_train_resampled), axis=1)

print(f'Starting size: {x_train.shape}   =>  Ovesampled Training size: {x_train_resampled.shape}')


# In[33]:


from sklearn.preprocessing import StandardScaler

scl = StandardScaler()

x_train_scaled = scl.fit_transform(x_train_resampled)
x_test_scaled = scl.transform(x_test)

print('No change in dimension')


# In[34]:


from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(handle_unknown='ignore')


# In[36]:


x_train_scaled.shape


# In[37]:


x_train


# In[40]:


x_train_scaled[:, -1]


# In[ ]:




