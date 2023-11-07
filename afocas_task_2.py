# -*- coding: utf-8 -*-
"""AFOCAS_TASK_2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1bEjZJG1UMdI_mncn4qz2S_EMNaV5bJNI

### **Importing the Important libraries**
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""## **Importing the dataset and exploring it**"""

data=pd.read_csv("/content/iris.data")

data.info()

data.head()

data.tail()

"""## **Giving the columns or features names as they do not exist in the given dataset**"""

column_names = ['feature_1', 'feature_2', 'feature_3','feature_4','name']
data.columns = column_names

data.head()

data.info()

data.describe()

"""## **Checking how many types of outputs are there and then encoding them into numbers**"""

unique_values = data['name'].unique()
num_unique_values = len(unique_values)

print(f"Number of unique categorical values in '{'name'}': {num_unique_values}")

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
data['name'] = label_encoder.fit_transform(data['name'])

data.head()

data.tail()

"""## **Shuffling the dataset randomly**"""

data = data.sample(frac=1, random_state=42)

data.head()

"""## **Splitting the dataset into train and test**"""

from sklearn.model_selection import train_test_split

X = data.drop('name', axis=1)
y = data['name']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""# **Applying the logostic regression model**"""

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

logistic_reg_model = LogisticRegression()
logistic_reg_model.fit(X_train, y_train)

y_pred = logistic_reg_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

conf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(conf_matrix)

class_report = classification_report(y_test, y_pred)
print('Classification Report:')
print(class_report)

"""# **Applying the decision tree model**"""

from sklearn.tree import DecisionTreeClassifier

decision_tree_model = DecisionTreeClassifier(random_state=42)

decision_tree_model.fit(X_train, y_train)

y_pred_decision_tree = decision_tree_model.predict(X_test)

accuracy_decision_tree = accuracy_score(y_test, y_pred_decision_tree)
print(f'Decision Tree Accuracy: {accuracy_decision_tree:.2f}')

"""# **Applying the random forest model**"""

from sklearn.ensemble import RandomForestClassifier

random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)

random_forest_model.fit(X_train, y_train)

y_pred_random_forest = random_forest_model.predict(X_test)

accuracy_random_forest = accuracy_score(y_test, y_pred_random_forest)
print(f'Random Forest Accuracy: {accuracy_random_forest:.2f}')

"""# **Applying the Support vector machine model**"""

from sklearn.svm import SVC

svm_model = SVC(kernel='linear', C=1, random_state=42)

svm_model.fit(X_train, y_train)

y_pred_svm = svm_model.predict(X_test)

accuracy_svm = accuracy_score(y_test, y_pred_svm)
print(f'SVM Accuracy: {accuracy_svm:.2f}')

