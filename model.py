# -*- coding: utf-8 -*-
"""short.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1KmhAsqu-d3Yu5Pbif-F9nHIhQUgqRTkR
"""

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

#Import the dataset
df = pd.read_csv('Tel_Aviv.csv')
df['datetime'] = pd.DatetimeIndex(df.datetime)
df.head()

"""### Dealing with Missing Data"""

#Checking out total null values
df.isnull().sum()

df[df['temperature'].isna()]

#Since we have 793 values of fully empty rows and it is less than 5% of original data, we will remove them from dataset
df.drop(df[df['temperature'].isna()].index, inplace=True)
df

df.isnull().sum()

#Filling in missing values by median
cols = ['humidity',	'pressure', 'wind speed']
df[cols] = df[cols].fillna(df.median().iloc[0])
df

df.isnull().sum()

#Incorporating the year, month, day and hour into dataset for diversity
df['datetime'] = pd.DatetimeIndex(df.datetime)
df['date'] = df.datetime.dt.date
df['year'] = df.datetime.dt.year
df['month'] = df.datetime.dt.month
df['day'] = df.datetime.dt.day
df['hour'] = df.datetime.dt.hour

#Averaging the original feature set according to date
df['avg_humidity'] = df.groupby('date')['humidity'].transform('mean')
df['avg_pressure'] = df.groupby('date')['pressure'].transform('mean')
df['avg_temperature'] = df.groupby('date')['temperature'].transform('mean')
df['avg_wind_direction'] = df.groupby('date')['wind direction'].transform('mean')
df['avg_wind_speed'] = df.groupby('date')['wind speed'].transform('mean')
df

#Code for making the target variable into just 2 categories
df['weather'] = ['sky is clear' if i == 'sky is clear' else 'sky is not clear' for i in df['weather']]
df.tail(20)

df.drop(['datetime', 'date'], axis = 1, inplace = True)
df.head()

"""## Modelling"""

#Splitting into train and test dataset
features = df.drop('weather', axis=1)
target = df['weather']


x_train, x_val, y_train, y_val = train_test_split(features, target, test_size=0.3, random_state=100)

#Standardising the values 
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_val = sc.transform(x_val)

#Initialising a dictionary which stores all metrics
metrics_dict = dict()
model_names = ['Random Forest']

def cross_val_report(model, X,Y, report=True, save=False):
  
  '''
  A Small function to do cross validation as well as printing Classification report
  Arguments():
  model:
  The Classifier Model that was created with different ML Algorithms
  X : list/tuple 
  Contains both x_train and x_test values
  Y : list/tuple
  Contains both y_train and y_test values
  report: bool, default=True
  To print a classification report on testing data as well as a confusion matrix
  save: bool, default=False
  To save the model that is passed onto the function
  
  return: list
  A list containing various scores which is to be used for further analysis
  
  '''
        
  x_train, x_val = X[0], X[1]
  y_train, y_val = Y[0], Y[1]
  CV = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=100)
  n_scores = cross_val_score(model, x_train, y_train, scoring='accuracy', cv=CV, n_jobs=-1, error_score='raise')
  print('Accuracy: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))

  model.fit(x_train, y_train)
  y_pred = model.predict(x_val)
  clf_metrics = [accuracy_score(y_val, y_pred), np.mean(precision_score(y_val, y_pred, average=None)), np.mean(recall_score(y_val, y_pred, average=None)), np.mean(f1_score(y_val, y_pred, average=None))]

  if save:
    with open('model.pkl', 'wb') as files:
      pickle.dump(model, files)

  if report:
    print(classification_report(y_val, y_pred, zero_division=0))
    conf_mat = confusion_matrix(y_val, y_pred, labels=model.classes_)
    cm_display = ConfusionMatrixDisplay(conf_mat, display_labels=model.classes_)
    cm_display.plot()
    plt.show()
  return clf_metrics

rfc = RandomForestClassifier(n_estimators=150, criterion='entropy')
cross_val_report(rfc, [x_train, x_val], [y_train, y_val], save=True)