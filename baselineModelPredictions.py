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

#X data
features = path.expanduser(r'C:\Users\Nick Kim\Documents\Nicholas Kim\School\School Semesters\Spring 2019 Semester\Fundamentals of Machine Learning\Assignment 2\Fragile Families\criticalFeaturesFatherContinuous.csv')
#Y_train
labels = path.expanduser(r'C:\Users\Nick Kim\Documents\Nicholas Kim\School\School Semesters\Spring 2019 Semester\Fundamentals of Machine Learning\Assignment 2\Fragile Families\train.csv')
#Y_test
predictLabels = path.expanduser(r'C:\Users\Nick Kim\Documents\Nicholas Kim\School\School Semesters\Spring 2019 Semester\Fundamentals of Machine Learning\Assignment 2\Fragile Families\test.csv')


# Load training data, training labels, and test labels
Training = selectFeatures.selectDF(motherFatherAll)
labels = pd.read_csv(labels)
predictLabels = pd.read_csv(predictLabels)
# Scales all the features between range [0,1]
scaler = preprocessing.MinMaxScaler()
scaled_values = scaler.fit_transform(Training)
Training.loc[:,:] = scaled_values
# unscales the challengeID feature
for i in range(4242):
    Training['challengeID'][i] = int(i+1)

trainChallengeIDs = []
y_train = []
X_train = []

#1 = challenge ID, 3 = Grit
for row in labels.itertuples():
    trainChallengeIDs.append(row[1])
    y_train.append(row[3])

# Fills X_train
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
# Deletes challengeIDs with NA label for Grit
topIndex = len(y_train) - 1
for i in range (topIndex, -1, -1):
    if (math.isnan(y_train[i])):
        del y_train[i]
        del X_train[i]

testChallengeIDs = []
y_test = []
X_test = []

#1 = ChallengeID, 3 = Grit
for row in predictLabels.itertuples():
    testChallengeIDs.append(row[1])
    y_test.append(row[3])

#Fills X_test
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

#Removes challengeIDs with NA label for Grit
topIndex = len(y_test) - 1
for i in range (topIndex, -1, -1):
    if (math.isnan(y_test[i])):
        del y_test[i]
        del X_test[i]


# Uncomment as needed to run the desired model
model = LinearRegression()
#model = BayesianRidge()
#model = SVR()
#model = SGDRegressor()
#model = DecisionTreeRegressor()
#model = GradientBoostingRegressor()
#model = KNeighborsRegressor()
#model = MLPRegressor()
model.fit(X_train, y_train)

# Prints statistics
print(model)
print("R2:" + str(model.score(X_test, y_test)))
print("Mean Squared Error: " + str(mean_squared_error(y_test, model.predict(X_test))))
print("Mean Absolute Error: " + str(mean_absolute_error(y_test, model.predict(X_test))))
print("Median Absolute Error: " + str(median_absolute_error(y_test, model.predict(X_test))))
# Test to see MSE of using mean value for every prediction
y_mean = []
val = mean(y_test)
for i in range(len(y_test)):
    y_mean.append(val)
#Statistics of mean prediction
print("Mean Value Prediction")
print("R2: " + str(0.0))
print("Mean Squared Error: " + str(mean_squared_error(y_test, y_mean)))
print("Mean Absolute Error: " + str(mean_absolute_error(y_test, y_mean)))
print("Median Absolute Error: " + str(median_absolute_error(y_test, y_mean)))
