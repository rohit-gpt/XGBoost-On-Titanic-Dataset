#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 23:30:48 2019

@author: rohitgupta
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("train.csv")

dataset.drop('Name', inplace=True, axis=1)
dataset.drop('Cabin', inplace=True, axis=1)
dataset.drop('Ticket', inplace=True, axis=1)

dataset.isnull().sum()
dataset[dataset['Embarked'].isnull()]

dataset.drop([61, 829], inplace=True, axis = 0)

dataset.mean()
dataset.fillna(dataset.mean(), inplace=True)

X_train = dataset.iloc[:, :8].values
Y_train = dataset.iloc[:, 8].values

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
X_train[:,2] = label_encoder.fit_transform(X_train[:,2])
X_train[:,7] = label_encoder.fit_transform(X_train[:,7])

from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features=[7])
X_train = onehotencoder.fit_transform(X_train).toarray()

X_train = X_train[:, 1:]

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)

X_train = X_train[:, 1:]

import statsmodels.formula.api as sm
X_train = np.append(arr=np.ones((889,1)).astype(int), values = X_train, axis=1)

X_opt = X_train[:, [0,2,4,5,6,7]]
classifier_OLS = sm.OLS(endog=Y_train, exog=X_opt).fit()
classifier_OLS.summary()





from xgboost import XGBClassifier
classifier = XGBClassifier(n_estimators=120)
classifier.fit(X_opt, Y_train)

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=classifier, X=X_opt, y=Y_train, cv=15)
accuracies.mean()
accuracies.std()






from sklearn.model_selection import GridSearchCV
parameters = [
        {'max_depth': [2,3,4], 'learning_rate': [0.08, 0.1, 0.12, 0.2], 'n_estimators': [50, 80, 95, 100, 120, 150]}
    ]
grid_search = GridSearchCV(estimator=classifier, param_grid=parameters, scoring='accuracy', cv=10)
grid_search = grid_search.fit(X_opt, Y_train)

best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_


testdata = pd.read_csv("test.csv")
testdata.drop('Name', axis = 1, inplace=True)
testdata.drop('Ticket', axis = 1, inplace=True)
testdata.drop('Cabin', axis = 1, inplace=True)
testdata.isnull().sum()
testdata.fillna(testdata.mean(), inplace=True)
X_test = testdata.iloc[:,:].values
X_test[:,2] = label_encoder.fit_transform(X_test[:,2])
X_test[:,7] = label_encoder.fit_transform(X_test[:,7])
X_test = onehotencoder.fit_transform(X_test).toarray()
X_test = X_test[:,1:]
X_test = np.append(arr=np.ones((418,1)).astype(int), values = X_test, axis=1)
X_test = sc.transform(X_test)

X_test = X_test[:, [0,2,4,5,6,7]]

y_pred = classifier.predict(X_test)
pid = testdata.iloc[:,0].values
a = np.column_stack((pid, y_pred))

np.savetxt("results.csv", a, delimiter=",")