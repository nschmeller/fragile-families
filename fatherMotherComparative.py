import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import selectFeatures
from selectFeatures import motherB, mother1, mother3, mother5, mother9, fatherB, father1, father3, father5, father9
from os import path
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error
from sklearn import preprocessing
from statistics import mean

#y_train
labels = path.expanduser(r'C:\Users\Nick Kim\Documents\Nicholas Kim\School\School Semesters\Spring 2019 Semester\Fundamentals of Machine Learning\Assignment 2\Fragile Families\train.csv')
#y_test
predictLabels = path.expanduser(r'C:\Users\Nick Kim\Documents\Nicholas Kim\School\School Semesters\Spring 2019 Semester\Fundamentals of Machine Learning\Assignment 2\Fragile Families\test.csv')

#List of lists that contain feature names for each wave and mother or father
dataNames = [motherB, mother1, mother3, mother5, mother9, fatherB, father1, father3, father5, father9]
dataNamesStrings = ['motherB', 'mother1', 'mother3', 'mother5', 'mother9', 'fatherB', 'father1', 'father3', 'father5', 'father9']
#Dataframe for plotting
df = pd.DataFrame(columns = ['Features', 'Bayesian Ridge', 'SVR', 'Predict Mean for All'])
df['Features'] = dataNamesStrings

#Main loop
for index, data in enumerate(dataNames):
    print(dataNamesStrings[index])
    # Load training data
    Training = selectFeatures.selectDF(data)
    labels = pd.read_csv(path.expanduser(r'C:\Users\Nick Kim\Documents\Nicholas Kim\School\School Semesters\Spring 2019 Semester\Fundamentals of Machine Learning\Assignment 2\Fragile Families\train.csv'))
    predictLabels = pd.read_csv(path.expanduser(r'C:\Users\Nick Kim\Documents\Nicholas Kim\School\School Semesters\Spring 2019 Semester\Fundamentals of Machine Learning\Assignment 2\Fragile Families\test.csv'))
    # Scales data
    scaler = preprocessing.MinMaxScaler()
    scaled_values = scaler.fit_transform(Training)
    Training.loc[:,:] = scaled_values
    for i in range(4242):
        Training['challengeID'][i] = int(i+1)

    trainChallengeIDs = []
    y_train = []
    X_train = []

    #1, challenge ID, #3, Grit
    for row in labels.itertuples():
        trainChallengeIDs.append(row[1])
        y_train.append(row[3])


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
    model = BayesianRidge()
    model2 = SVR()

    model.fit(X_train, y_train)
    model2.fit(X_train, y_train)

    print("Bayesian Ridge")
    print("R2:" + str(model.score(X_test, y_test)))
    print("Mean Squared Error: " + str(mean_squared_error(y_test, model.predict(X_test))))
    df['Bayesian Ridge'][index] = mean_squared_error(y_test, model.predict(X_test))
    print("Mean Absolute Error: " + str(mean_absolute_error(y_test, model.predict(X_test))))
    print("Median Absolute Error: " + str(median_absolute_error(y_test, model.predict(X_test))))

    print("")
    print("Support Vector Regression")
    print("R2:" + str(model2.score(X_test, y_test)))
    print("Mean Squared Error: " + str(mean_squared_error(y_test, model2.predict(X_test))))
    df['SVR'][index] = mean_squared_error(y_test, model2.predict(X_test))
    print("Mean Absolute Error: " + str(mean_absolute_error(y_test, model2.predict(X_test))))
    print("Median Absolute Error: " + str(median_absolute_error(y_test, model2.predict(X_test))))

    # Test to see MSE of using mean value for every prediction
    y_mean = []
    val = mean(y_test)
    for i in range(len(y_test)):
        y_mean.append(val)
    #print(mean_squared_error(y_test,y_mean))
    print(" ")
    print("")
    print("Mean Value Prediction")
    print("R2: " + str(0.0))
    print("Mean Squared Error: " + str(mean_squared_error(y_test, y_mean)))
    df['Predict Mean for All'][index] = mean_squared_error(y_test, y_mean)
    print("Mean Absolute Error: " + str(mean_absolute_error(y_test, y_mean)))
    print("Median Absolute Error: " + str(median_absolute_error(y_test, y_mean)))
    print("______________________________")


# Creates a bar plot of the comparative information
font = {'family': 'Calibri',
        'color':  'black',
        'weight': 'bold',
        'size': 14,
        }

print(df.head)

ax = df.plot.bar(rot=0)
ax.set_xticks(df.index)
ax.set_xticklabels(df.Features)
plt.ylabel("Mean Squared Error", fontdict = font)
plt.xlabel("Selected Features", fontdict = font)
plt.title("Grit Prediction Accuracy: Mother/Father Specific Years", fontweight = "bold")
plt.show()
