#%%
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
#%% md
## Visualization
#%%
df = pd.read_csv('/users/jorge/desktop/data visualization/train.csv')
df.head()
#%%
print('data shape:', df.shape)
print('data dim:', df.ndim)
print(type(df))
#%% md
### data information
#%%
df.info()
#%%
df.isnull().sum() #Checking Null values
#%% md
### Scatterplot using seaborn
#%%
sns.scatterplot(data=df, x='Age', y='Fare', hue='Sex')
#%% md
### chart pie using matplotlib
#%%
plt.style.use('seaborn')
df.groupby(['Sex']).sum().plot(kind= 'pie', y= 'PassengerId', shadow = True, autopct='%1.1f%%')
#%% md
### survival passengers
#%%
sns.catplot(x='Pclass', hue='Sex', col='Survived', # 0 = No, 1 = Yes
            data=df, kind='count',
            height=4, aspect=.7);
#%% md
### Which gender had a better chance of survival
#%%
sns.countplot(y='Survived', hue='Sex', data=df)
#We can see clearly by visualization that females had a better chance of survival
#%% md
### survival rate #siblings / Spouse
#%%
fig = plt.figure()
ax = sns.countplot(x = 'SibSp', hue = 'Survived', data = df)
ax.set_title('Survival Rate with Total of Siblings and Spouse')
ax.set_xlabel('Sibling and Spouse')
ax.set_ylabel('Count')
ax.legend(['No','Yes'],loc = 1)
#%% md
### survival rate #parents / children
#%%
fig = plt.figure()
ax = sns.countplot(x = 'Parch', hue = 'Survived', data = df)
ax.set_title('Survival Rate with Total Parents and Children')
ax.set_xlabel('Parents and Children')
ax.set_ylabel('Count')
ax.legend(['No','Yes'],loc = 1)
#%% md
### survival sorted by class
#%%
df.groupby('Pclass').Survived.mean()
#%%
fig = plt.figure(figsize=(15,8),)
sns.kdeplot(df.Pclass[df.Survived == 0] ,
               color='grey',
               shade=True,
               label='not survived')
sns.kdeplot(df.loc[(df['Survived'] == 1),'Pclass'] ,
               color='blue',
               shade=True,
               label='survived',
              )
plt.title('Passenger Class - Survived / Non-Survivor', fontsize = 30)
plt.ylabel("Frequency", fontsize = 25, labelpad = 35)
plt.xlabel("Passenger Class", fontsize = 25,labelpad =25)
labels = ['1st Class', '2nd Class', '3rd Class']
plt.xticks(sorted(df.Pclass.unique()), labels, fontsize = 25);
#%% md
### survival sorted by age
#%%
fig = plt.figure(figsize=(15,8),)
sns.kdeplot(df.Age[df.Survived == 0] ,
               color='grey',
               shade=True,
               label='not survived')
sns.kdeplot(df.loc[(df['Survived'] == 1),'Age'] ,
               color='blue',
               shade=True,
               label='survived',
              )
plt.title('Passenger Age - Survived / Non-Survivor', fontsize = 30)
plt.ylabel("Frequency", fontsize = 25, labelpad = 20)
plt.xlabel("Passenger Age", fontsize = 25,labelpad =20)
