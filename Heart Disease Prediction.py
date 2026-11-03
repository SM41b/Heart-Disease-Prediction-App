#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing necessary libraries for data processing, model training, and evaluation
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib


# In[2]:


#loading csv to pandas dataframe
data = pd.read_csv('heart.csv')
data.head()


# In[3]:


#number of rows and columns in dataset
data.shape


# In[4]:


#dataset information
data.info()


# In[5]:


#checking for missing values
data.isnull().sum()


# In[6]:


#statistical measures of data
data.describe()


# In[7]:


#checking the distribution of target values
data['target'].value_counts()


# In[8]:


# Separating independent features and dependent target
X = data.drop(columns='target')
Y = data['target']


# In[9]:


print(X)


# In[10]:


print(Y)


# In[11]:


# Splitting the original dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=1, stratify=Y
)


# In[12]:


print(X.shape , X_train.shape , X_test.shape)


# In[13]:


# Creating a pipeline to scale features and train the logistic regression model
model = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression())
])

model.fit(X_train, Y_train)


# In[14]:


# Evaluating model accuracy on training data
X_train_pred = model.predict(X_train)
train_data_accuracy = accuracy_score(Y_train, X_train_pred)
print("Accuracy on training data:", train_data_accuracy)


# In[15]:


# Evaluating model accuracy on unseen testing data
X_test_pred = model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_pred)
print("Accuracy on testing data:", test_data_accuracy)


# In[16]:


# Displaying precision, recall, and F1-score for each target class
print(classification_report(Y_test, X_test_pred))


# In[17]:


# Displaying correct and incorrect predictions made by the model
print(confusion_matrix(Y_test, X_test_pred))


# In[18]:


# Predicting heart disease using new patient input
input_data = (54, 1, 2, 120, 258, 0, 0, 147, 0, 0.4, 1, 0, 3)

input_df = pd.DataFrame([input_data], columns=X.columns)
prediction = model.predict(input_df)

if prediction[0] == 0:
    print("The person does not have heart disease.")
else:
    print("The person has heart disease.")


# In[19]:


# Saving the complete machine learning pipeline for future use
joblib.dump(model, 'heart_disease_model.pkl')



