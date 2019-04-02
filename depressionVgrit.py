import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import selectFeatures
from selectFeatures import motherFatherAll
from os import path
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error
from sklearn import preprocessing
from statistics import mean

#Loads depression features created by depressionVgritFeatureGenerator.py
features = path.expanduser(r'C:\Users\Nick Kim\Documents\Nicholas Kim\School\School Semesters\Spring 2019 Semester\Fundamentals of Machine Learning\Assignment 2\Fragile Families\depressionFeatures.csv')
labels = path.expanduser(r'C:\Users\Nick Kim\Documents\Nicholas Kim\School\School Semesters\Spring 2019 Semester\Fundamentals of Machine Learning\Assignment 2\Fragile Families\train.csv')
predictLabels = path.expanduser(r'C:\Users\Nick Kim\Documents\Nicholas Kim\School\School Semesters\Spring 2019 Semester\Fundamentals of Machine Learning\Assignment 2\Fragile Families\test.csv')


#want to decrease loading times
#Write out a csv that only contains the columns from the relevant categories


# Load training data
Training = pd.read_csv(features)
labels = pd.read_csv(labels)
predictLabels = pd.read_csv(predictLabels)

Training = Training.drop(columns = ['cf2md_case_con', 'cf2md_case_lib', 'cm2md_case_con', 'cm2md_case_lib', 'cf3md_case_con', 'cf3md_case_lib', 'cm3md_case_con', 'cm3md_case_lib', 'cf4md_case_con', 'cf4md_case_lib', 'cm4md_case_con', 'cm4md_case_lib', 'cf5md_case_con', 'cf5md_case_lib', 'cm5md_case_con', 'cm5md_case_lib'])


trainChallengeIDs = []
y_train = []
X_train = []

#1, challenge ID, #3, Grit
for row in labels.itertuples():
    trainChallengeIDs.append(row[1])
    y_train.append(row[3])

print(y_train)


IDIndex = 0
for row in Training.itertuples():
    if (IDIndex >= len(trainChallengeIDs)):
        break
    if (row[1] != trainChallengeIDs[IDIndex]):
        continue
    IDIndex += 1
    newFeatureVector = []
    for i in range(2, len(row)):
        newFeatureVector.append(row[i])
    X_train.append(newFeatureVector)

topIndex = len(y_train) - 1
for i in range (topIndex, -1, -1):
    if (math.isnan(y_train[i])):
        del y_train[i]
        del X_train[i]

#X_train and y_train finished
testChallengeIDs = []
y_test = []
X_test = []

#1, challenge ID, #3, Grit
for row in predictLabels.itertuples():
    testChallengeIDs.append(row[1])
    y_test.append(row[3])

IDIndex = 0
for row in Training.itertuples():
    if (IDIndex >= len(testChallengeIDs)):
        break
    if (row[1] != testChallengeIDs[IDIndex]):
        continue
    IDIndex += 1
    newFeatureVector = []
    for i in range(2, len(row)):
        newFeatureVector.append(row[i])
    X_test.append(newFeatureVector)

topIndex = len(y_test) - 1
for i in range (topIndex, -1, -1):
    if (math.isnan(y_test[i])):
        del y_test[i]
        del X_test[i]


# CHOOSING THE MODEL
model = LinearRegression()
model.fit(X_train, y_train)

X = []
for val in X_train:
    for i in val:
        X.append(i)
for val in X_test:
    for i in val:
        X.append(i)

y = []
for val in y_train:
    y.append(val)
for val in y_test:
    y.append(val)

from scipy.stats.stats import pearsonr
print(pearsonr(X,y))
print('_______')

print(model)
print("R2:" + str(model.score(X_test, y_test)))
print("Mean Squared Error: " + str(mean_squared_error(y_test, model.predict(X_test))))
print("Mean Absolute Error: " + str(mean_absolute_error(y_test, model.predict(X_test))))
print("Median Absolute Error: " + str(median_absolute_error(y_test, model.predict(X_test))))
