#!/usr/bin/env python
# coding: utf-8

# # Cities

# In[2]:


from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np


# In[5]:


df=pd.read_csv(r"G:\COVID-19 Project\coronacity.csv")


# In[6]:


df.head(2)


#  # Data Analysis

# In[7]:


xpos = np.arange(len(df.Division))
xpos


# In[8]:


plt.bar(df.Division,df.d_17_5_2020,color='black',)
plt.xlabel('Divisons')
plt.ylabel('Cases')
plt.title('Divison Cases')
plt.tick_params(axis='x', rotation=80)


# In[9]:


from sklearn import preprocessing


# In[10]:


label_encoder = preprocessing.LabelEncoder() 


# In[11]:


df['Cities']= label_encoder.fit_transform(df['Cities']) 


# In[12]:


df['Cities'].unique() 


# # K means Algorithm

# In[13]:


plt.scatter(df.d_17_5_2020,df['Cities'])
plt.xlabel('Cities')
plt.ylabel('d_29_4_20')
plt.title('Cities Condition')


# In[14]:


km = KMeans(n_clusters=4)
y_predicted = km.fit_predict(df[['d_17_5_2020']])
y_predicted


# In[15]:


df['cluster']=y_predicted
df.head(2)


# In[18]:


df1 = df[df.cluster==0]
df2 = df[df.cluster==1]
df3 = df[df.cluster==2]
df4 = df[df.cluster==3]


plt.scatter(df1.Cities,df1['d_17_5_2020'],color='green')
plt.scatter(df2.Cities,df2['d_17_5_2020'],color='red')
plt.scatter(df3.Cities,df3['d_17_5_2020'],color='blue')
plt.scatter(df4.Cities,df4['d_17_5_2020'],color='yellow')

plt.xlabel('Cities')
plt.ylabel('Cases')
plt.title('Situation Of Cities')
plt.legend()


# In[19]:


df1 = df[df.cluster==0]
df2 = df[df.cluster==1]
df3 = df[df.cluster==2]
df4 = df[df.cluster==3]


plt.scatter(df1.d_17_5_2020,df1['Cities'],color='green')
plt.scatter(df2.d_17_5_2020,df2['Cities'],color='red')
plt.scatter(df3.d_17_5_2020,df3['Cities'],color='blue')
plt.scatter(df4.d_17_5_2020,df4['Cities'],color='yellow')



plt.xlabel('Cities')
plt.ylabel('Cases')
plt.title('Situation Of Cities')
plt.legend()


# # Dhaka Cities 

# In[20]:


from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np


# In[22]:


df=pd.read_csv(r"G:\COVID-19 Project\dhakacitydata.csv")


# In[23]:


df.head(2)


# # Data Analysis

# In[24]:


xpos = np.arange(len(df.Cities))
xpos


# In[25]:


plt.bar(df.Cities,df.d_15_5_2020,color='black',)
plt.xlabel('Cities')
plt.ylabel('Cases')
plt.title('Cities Cases')


# In[26]:


from sklearn import preprocessing


# In[27]:


label_encoder = preprocessing.LabelEncoder()


# In[28]:


df['Cities']= label_encoder.fit_transform(df['Cities'])


# In[29]:


df['Cities'].unique()


# # K means  Algorithm

# In[31]:


plt.scatter(df.Cities,df['d_17_5_2020'])
plt.xlabel('Cities')
plt.ylabel('d_17_5_2020')


# In[32]:


km = KMeans(n_clusters=3)
y_predicted = km.fit_predict(df[['d_17_5_2020']])
y_predicted


# In[33]:


df['cluster']=y_predicted
df.head(2)


# In[34]:


df1 = df[df.cluster==0]
df2 = df[df.cluster==1]
df3 = df[df.cluster==2]


plt.scatter(df1.Cities,df1['d_17_5_2020'],color='green')
plt.scatter(df2.Cities,df2['d_17_5_2020'],color='red')
plt.scatter(df3.Cities,df3['d_17_5_2020'],color='yellow')


plt.xlabel('Cities')
plt.ylabel('Cases')
plt.title('Situation Of Cities')
plt.legend()


# # Agglomerative Clustering

# In[35]:


plt.scatter(df.d_21_4_20,df['d_17_5_2020'])
plt.xlabel('Cities')
plt.ylabel('d_29_4_20')
plt.title("Situation of Cities")


# In[36]:


km = AgglomerativeClustering(n_clusters=3)
y_predicted = km.fit_predict(df[['d_17_5_2020']])
y_predicted


# In[37]:


df['cluster']=y_predicted
df.head(2)


# In[38]:


df1 = df[df.cluster==0]
df2 = df[df.cluster==1]
df3= df[df.cluster==2]
plt.scatter(df1.d_21_4_20,df1['d_17_5_2020'],color='red')
plt.scatter(df2.d_21_4_20,df2['d_17_5_2020'],color='yellow')
plt.scatter(df3.d_21_4_20,df3['d_17_5_2020'],color='green')
plt.xlabel('Cities')
plt.ylabel('Cases')
plt.title('Situation Of Cities')
plt.legend()


#  # Date 
# 

# In[39]:


df=pd.read_csv(r"G:\COVID-19 Project\datasetdate.csv")


# In[40]:


df.head(2)


# # Data Analysis 

# In[41]:


plt.bar(df.date,df.f_d,color='Red',)
plt.xlabel('Date')
plt.ylabel('')
plt.title('Female Death Cases')
plt.tick_params(axis='x', rotation=90)


# In[51]:


plt.bar(df.date,df.m_d,color='Red',)
plt.xlabel('Date')
plt.ylabel('')
plt.title('Male Death Cases')
plt.tick_params(axis='x', rotation=90)


# In[52]:


plt.bar(df.date,df.dha_d,color='Red',)
plt.xlabel('Date')
plt.ylabel('')
plt.title('Dhaka Death Cases')
plt.tick_params(axis='x', rotation=90)


# In[53]:


plt.bar(df.date,df.syl_d,color='Red',)
plt.xlabel('Days')
plt.ylabel('')
plt.title('Sylet Death Cases')
plt.tick_params(axis='x', rotation=80)


# In[54]:


plt.xlabel('Days')
plt.ylabel('')
plt.title('Percentage Of Male and Female cases Analysis')


plt.plot(df.date, df.c_m_p, label="Male Cases")
plt.plot(df.date, df.c_f_p, label="Female Cases")

plt.legend(loc='best')
plt.tick_params(axis='x', rotation=80)


# In[55]:


y = df[['c_a_p_30']]
x = df.drop(['c_f_p','c_m_p','a_31_40','dha_h_q','syl_h_q','bar_h_q','ran_h_q','chi_h_q','raj_h_q','khu_h_q','dha_c','syl_c','bar_c','ran_c','chi_c','raj_c','khu_c','dha_d','syl_d','bar_d','ran_d','chi_d','raj_d','khu_d','f_d','m_d','a_0_10','a_11_20','a_21_30','a_41_50','a_51_60','a_60','t_q','c_a_p_10','c_a_p_20','c_a_p_40','c_a_p_30','c_a_p_50','c_a_p_60','c_a_p_60+'],axis=1)
x=x.dropna()


# In[56]:


from sklearn.preprocessing import LabelEncoder


# In[57]:


le=LabelEncoder()


# In[58]:


le.fit(x['date'])


# In[59]:


x['date']=le.transform(x['date'])


#  # Linear Regression Model 

# In[60]:


from sklearn.model_selection import train_test_split


# In[61]:


x_train, x_test, y_train, y_test = train_test_split(
x, y, test_size=0.2, random_state=10)


# In[62]:


from sklearn.linear_model import LinearRegression


# In[63]:


reg=LinearRegression()


# In[64]:


reg.fit(x_train, y_train)


# In[65]:


reg.predict([[100]])


# # k-nearest neighbors algorithm

# In[66]:


from sklearn.neighbors import KNeighborsRegressor


# In[67]:


neigh = KNeighborsRegressor(n_neighbors=2)


# In[68]:


neigh.fit(x_train, y_train)


# In[69]:


neigh.predict([[100]])


# In[70]:


neigh.score(x_test,y_test)


# # The End

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # Data Analysis

# # Data Vizulaization

# # Data cleaning
# 

# # Machine learning 

# #  Unsupervised Learning by using multiple libraries in Python.

# # Machine Learning for Data Science

# # Tool : Python

# In[ ]:





# In[ ]:




