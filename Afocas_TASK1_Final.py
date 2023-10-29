#!/usr/bin/env python
# coding: utf-8

# # Car Prediction Model

# ## Importing libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib as plt


# ## Loading the dataset

# In[2]:


train_data=pd.read_csv("train.csv")
test_data=pd.read_csv("test.csv")


# ## Exploring the dataset

# In[3]:


train_data.info()


# In[4]:


train_data.head()


# ## Converting the Year feature to the age of the car till date

# In[5]:


train_data['Prod. year'] = pd.to_datetime(train_data['Prod. year'], format='%Y').dt.year

current_year = 2023 
train_data['Car Age'] = current_year - train_data['Prod. year']


# In[6]:


test_data['Prod. year'] = pd.to_datetime(test_data['Prod. year'], format='%Y').dt.year

current_year = 2023  
test_data['Car Age'] = current_year - test_data['Prod. year']


# In[7]:


train_data.drop(columns=['Prod. year'],axis=1,inplace=True)


# In[8]:


test_data.drop(columns=['Prod. year'],axis=1,inplace=True)


# In[9]:


train_data.head()


# In[10]:


train_data.describe()


# ## Remove 'km' from the 'Mileage' feature and convert to numeric

# In[11]:


train_data['Mileage'] = train_data['Mileage'].str.replace(' km', '', regex=False).astype(int)


# In[12]:


test_data['Mileage'] = test_data['Mileage'].str.replace(' km', '', regex=False).astype(int)


# ## Converting feature levy to numeric values

# In[13]:


train_data['Levy'] = pd.to_numeric(train_data['Levy'], errors='coerce')


# In[14]:


test_data['Levy'] = pd.to_numeric(test_data['Levy'], errors='coerce') 


# In[15]:


train_data.isnull().sum()


# ## As there are many null values in levy so dropping the column

# In[16]:


train_data.drop(columns=['Levy'],axis=1,inplace=True)


# In[17]:


test_data.drop(columns=['Levy'],axis=1,inplace=True)


# ## Removig the string turbo from the feature Engine volume and making it a separate feature also making the Engine volume as an int 

# In[18]:


train_data['Turbo'] = train_data['Engine volume'].str.contains('Turbo', case=False).astype(int)
train_data['Engine volume'] = train_data['Engine volume'].str.extract(r'(\d+\.\d+|\d+)').astype(float)


# In[19]:


test_data['Turbo'] = test_data['Engine volume'].str.contains('Turbo', case=False).astype(int)
test_data['Engine volume'] = test_data['Engine volume'].str.extract(r'(\d+\.\d+|\d+)').astype(float)


# ## Dropping the irrelevent columns 

# In[20]:


train_data.drop(columns=['ID'],axis=1,inplace=True)


# In[21]:


test_data.drop(columns=['ID'],axis=1,inplace=True)


# In[22]:


train_data.drop(columns=['Model'],axis=1,inplace=True)


# In[23]:


test_data.drop(columns=['Model'],axis=1,inplace=True)


# In[24]:


train_data.info()


# ## Applying one hot encoding  to categorical values

# In[25]:


train_data = pd.get_dummies(train_data, drop_first=True)


# In[26]:


test_data = pd.get_dummies(test_data, drop_first=True)


# In[27]:


train_data.head()


# In[28]:


feature_names = train_data.columns
print(feature_names)


# In[29]:


test_data.head()


# In[30]:


test_data.head()


# ## Getting the columns which are not present in one but present in the other

# In[31]:


columns_df1 = set(train_data.columns)
columns_df2 = set(test_data.columns)

different_columns = columns_df1.symmetric_difference(columns_df2)

print("Columns that are different:")
for column in different_columns:
    print(column)


# In[32]:


columns_df1 = set(train_data.columns)
columns_df2 = set(test_data.columns)

columns_only_in_df1 = columns_df1 - columns_df2
columns_only_in_df2 = columns_df2 - columns_df1

print("Columns only in df1:")
for column in columns_only_in_df1:
    print(f"{column} is in df1 but not in df2")

print("Columns only in df2:")
for column in columns_only_in_df2:
    print(f"{column} is in df2 but not in df1")


# ## Adding the columns which are not present in the respective dataframe

# In[33]:


new_column_names = ['Manufacturer_FOTON', 'Manufacturer_TATA', 'Manufacturer_MG']

for column_name in new_column_names:
    train_data[column_name] = 0


# In[34]:


new_column_names = ['Manufacturer_HAVAL', 'Manufacturer_LAMBORGHINI', 'Manufacturer_PONTIAC','Manufacturer_SEAT','Manufacturer_ROLLS-ROYCE','Manufacturer_LANCIA']

for column_name in new_column_names:
    test_data[column_name] = 0


# In[35]:


test_data.head()


# In[36]:


train_data.head()


# ## Making the columns in order of both test and train set

# In[37]:


training_column_order = train_data.columns
test_data = test_data[training_column_order]


# In[38]:


train_data.head()


# In[39]:


test_data.head()


# ## Separating the Input and output from the dataset

# In[40]:


X_train = train_data.iloc[:,1:]
y_train = train_data.iloc[:,0]


# In[41]:


X_test = test_data.iloc[:,1:]
y_test = test_data.iloc[:,0]


# In[42]:


y_test.head()


# ## Traning the model through Decision trees

# In[43]:


from sklearn.tree import DecisionTreeRegressor

model = DecisionTreeRegressor()
model.fit(X_train, y_train)


# ## Evaluating the model through Mean squared error and the R squared metrics

# In[44]:


from sklearn.metrics import mean_squared_error, r2_score
y_pred = model.predict(X_train)

mse = mean_squared_error(y_train, y_pred)
r2 = r2_score(y_train, y_pred)

print("Mean Squared Error (MSE):", mse)
print("R-squared (R2):", r2)


# ## Predicting the test set and saving it in the test.csv file

# In[45]:


predictions= model.predict(X_test)

print(predictions)


# In[46]:



df = pd.read_csv('test.csv')



df['Price'] = predictions


# In[47]:


df.head()


# In[48]:


df.to_csv('test.csv', index=False)


# In[ ]:




