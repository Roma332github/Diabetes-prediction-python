#!/usr/bin/env python
# coding: utf-8

# # DIABETES PREDICTION SYSTEM 👩🏻‍⚕🏥📈

# ![diabetes-digital_adobe.jpg](attachment:diabetes-digital_adobe.jpg)

# # Introduction to Diabetes Prediction System👩🏻‍💻📝
# 
# ## In today's world, diabetes has become a widespread health concern affecting millions of people globally. Detecting diabetes early is crucial for effective management and prevention of complications. The Diabetes Prediction System project aims to use Python programming and machine learning techniques to predict the likelihood of an individual developing diabetes based on certain health parameters. This project can potentially assist healthcare professionals and individuals in making informed decisions about preventive measures and treatments.
# 
# # The primary objectives of this project are 🌟📌:
# 
# ## Developing a Predictive Model📋 :
# ### Build a machine learning model that can predict the probability of an individual developing diabetes based on their health data.
# 
# ## Data Preprocessing 📝 :
# ### Collect and clean the dataset, handling missing values, and ensuring data quality before feeding it into the model.
# 
# ## Model Selection and Training 💻 :
# ### Explore different machine learning algorithms such as Logistic Regression, Decision Trees, Random Forest, or Neural Networks to identify the most accurate predictive model.
# 
# ## Evaluation and Validation  ♻️:
# ### Assess the performance of the model using appropriate evaluation metrics to ensure reliability and accuracy.
# 
# ## User-Friendly Interface ✨ : 
# ### Create an intuitive interface where users can input their health metrics and receive predictions in a user-friendly format.

# # Benefits and Impact 🏆
# 
# ## The Diabetes Prediction System project has the potential to:
# 
# ### Assist healthcare professionals in identifying individuals at high risk of diabetes for early intervention. Empower individuals to make proactive lifestyle choices based on personalized risk assessments. Serve as a learning tool to understand the application of machine learning in healthcare and disease prediction.

# ## Importing Libraries

# In[ ]:


#Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ### Pandas: Data analysis and manipulation library for working with structured data using Data Frame and Series.
# ### NumPy: Numerical computing library supporting large, multi-dimensional arrays and matrices, with high-level mathematical functions.
# ### Seaborn: Statistical data visualization library for creating attractive and informative graphics, based on Matplotlib.
# ### Matplotlib: Comprehensive plotting library providing interface for creating various plots like line, scatter, bar, and histograms 

# # Important Libraries for Prediction

# In[3]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# ### Train Test Split: Technique for splitting data into training and testing sets to assess model performance.
# ### Logistic Regression: Method for predicting the probability of a binary outcome using the logistic function.
# ### Accuracy: Metric measuring the proportion of correctly classified instances in a classification model.
# ### Sklearn: Python's Scikit-learn, a powerful machine learning library providing tools for data analysis and model building 

# # Loading the Dataset

# In[4]:


data = pd.read_csv("diabetes.csv")


# In[5]:


data


# # Checking for Missing Values

# In[7]:


data.isnull()


# In[8]:


data.isnull().sum()


# In[6]:


sns.heatmap(data.isnull())


# # Co relation matrix

# In[11]:


correlation = data.corr()
correlation


# In[12]:


print(correlation)


# ## Visualizing Correlation 

# In[13]:


sns.heatmap(correlation)


# # Trainning the Model with the help of Train Test Split

# ## Train Test Split

# In[18]:


from sklearn.model_selection import train_test_split
x= data.drop("Outcome", axis=1)
y=data['Outcome']
x_train, x_test , y_train , y_test =train_test_split(x,y,test_size=0.2)


# ### In X all the independent variables are stored , In Y the predictor variable(“OUTCOME”) is stored.
# ### Train-test split is a technique used in machine learning to assess model performance. It  divides the dataset into a training set and a testing set, with a 0.2 test size indicating that  20% of the data is used for testing and 80% for training. 

# ## Trainning the Model

# In[20]:


from sklearn.linear_model import LinearRegression
model = LogisticRegression(max_iter=1000)
model.fit(X, y)
model.fit(x_train,y_train)


# ### Fitting the X train and y train data into the variable called model.

# # Making Prediction

# In[21]:


prediction = model.predict(x_test)


# In[22]:


print(prediction)


# ### After training the model, predictions are made using the test data, which comprises 20% of the total dataset

# In[23]:


accuracy = accuracy_score(prediction,y_test)


# In[24]:


print(accuracy)


# ### The accuracy of the model is then calculated and determined.

# In[ ]:




