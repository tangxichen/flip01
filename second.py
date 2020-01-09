#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


train_data=pd.read_csv('F:\\BaiduNetdiskDownload\\titanic\\train.csv')
test_data=pd.read_csv('F:\\BaiduNetdiskDownload\\titanic\\test.csv')
#set_style( )是用来设置主题的  https://blog.csdn.net/sinat_23338865/article/details/80405567
sns.set_style('whitegrid')
train_data.head()


# In[3]:


train_data.info()
print('*'*40)
test_data.info()


# In[4]:


train_data['Survived'].value_counts().plot.pie(labeldistance=1.1,shadow=True)


# In[5]:


train_data.Embarked[train_data.Embarked.isnull()]=train_data.Embarked.dropna().mode().values
#mode应该是众数，就是频数最高的那个。示例里面1和2都出现了3次，是最频繁的，所以返回的是这两个数字。
train_data['Cabin']=train_data.Cabin.fillna('U0')


# In[6]:


from sklearn.ensemble import RandomForestRegressor as RFR
age_df=train_data[['Age','Survived','Fare','Parch','SibSp','Pclass']]
age_df_notnull=age_df.loc[(train_data['Age'].notnull())]
age_df_isnull=age_df.loc[(train_data['Age'].isnull())]
x=age_df_notnull.values[:,1:]
y=age_df_notnull.values[:,0]

rfr=RFR(n_estimators=1000,n_jobs=-1)
rfr.fit(x,y)
predictsAges=rfr.predict(age_df_isnull.values[:,1:])
train_data.loc[train_data['Age'].isnull(),['Age']]=predictsAges #loc(),https://blog.csdn.net/sushangchun/article/details/83514803


# In[7]:


train_data.info()


# In[8]:


print(train_data.groupby(['Sex','Survived'])['Survived'].count())
print("*"*40)
temp=train_data[['Sex','Survived']].groupby(['Sex']).mean()


# In[9]:


temp.plot.bar()


# In[10]:


print(train_data.groupby(['Pclass','Survived'])['Pclass'].count())
print("*"*40)
print(train_data[["Pclass",'Survived']].groupby(['Pclass']).mean())
temp1=train_data[["Pclass",'Survived']].groupby(['Pclass']).mean()


# In[11]:


temp1.plot.bar()


# In[12]:


print(train_data[['Sex','Pclass','Survived']].groupby(['Sex','Pclass']).mean())
print('/-\\'*12)
train_data[['Sex','Pclass','Survived']].groupby(['Sex','Pclass']).mean().plot.bar()


# In[13]:


print(train_data[['Sex','Pclass','Survived']].groupby(['Sex','Pclass']).count())


# In[26]:


fig,ax=plt.subplots(1,2,figsize=(20,5))

ax[0].set_yticks(range(0,110,10))
sns.violinplot("Pclass","Age",hue="Survived",data=train_data,split=True,ax=ax[0])
ax[0].set_title("Pclass & Age - Survived")
#.violinplot,它显示了定量数据在一个（或多个）分类变量的多个层次上的分布，这些分布可以进行比较
#https://www.jianshu.com/p/96977b9869ac
ax[1].set_yticks(range(0,110,10))
sns.violinplot("Sex","Age",hue="Survived",data=train_data,split=True,ax=ax[1])
ax[1].set_title("Sex & Age -Survived")
plt.show()


# In[34]:


plt.figure(figsize=(15,5))

plt.subplot(121)
train_data["Age"].hist(bins=100)
plt.xlabel('Age')
plt.ylabel('Num')

plt.subplot(122)
train_data.boxplot(column='Age',showfliers=False)
plt.show()


# In[40]:


facet=sns.FacetGrid(train_data,hue="Survived",aspect=4)
#先sns.FacetGrid画出轮廓 ,然后用map填充内容
#结构化图表可视化,https://blog.csdn.net/qq_42554007/article/details/82627231
facet.map(sns.kdeplot,'Age',shade=True)
facet.set(xlim=(0,train_data['Age'].max()))
facet.add_legend()


# In[66]:


fig,axes1=plt.subplots(1,1,figsize=(18,4))
train_data["Age_int"]=train_data["Age"].astype(int)
average_age=train_data[["Age_int","Survived"]].groupby(['Age_int'],as_index=False).mean()
sns.barplot(x="Age_int",y="Survived",data=train_data)


# In[69]:


bins=[0,12,18,65,100]
train_data["Age_group"]=pd.cut(train_data['Age'],bins)
by_age=train_data.groupby("Age_group")["Survived"].mean()
print(by_age)


# In[ ]:




