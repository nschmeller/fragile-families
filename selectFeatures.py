import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from os import path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


features = path.expanduser(r'C:\Users\Nick Kim\Documents\Nicholas Kim\School\School Semesters\Spring 2019 Semester\Fundamentals of Machine Learning\Assignment 2\Fragile Families\outputModified.csv')

#want to decrease loading times
#Write out a csv that only contains the columns from the relevant categories


#LIST OF VARIED FEATURES
motherFatherAll = ['challengeID', 'cf1age', 'cf1adult', 'cf1kids', 'cf1hhinc', 'cf1inpov', 'cm1age', 'cm1adult', 'cm1kids', 'cm1hhinc', 'cm1inpov', 'cf2age', 'cf2adult', 'cf2kids', 'cf2hhinc', 'cf2hhincb', 'cf2povco', 'cf2povcob', 'cm2age', 'cm2amrf', 'cm2alvf', 'cm2fbir', 'cm2hhinc', 'cm2povco', 'cf3age', 'cf3adult', 'cf3kids', 'cf3hhinc', 'cf3hhincb', 'cf3povco', 'cf3povcob', 'cm3age', 'cm3amrf', 'cm3alvf', 'cm3adult', 'cm3kids', 'cm3hhinc', 'cm3povco', 'cf4age', 'cf4adult', 'cf4kids', 'cf4hhinc', 'cf4hhincb', 'cf4povco', 'cf4povcob', 'cm4age', 'cm4amrf', 'cm4alvf', 'cm4adult', 'cm4kids', 'cm4hhinc', 'cm4povco', 'cm5adult', 'cm5kids', 'cf5hhsize', 'cf5adult', 'cf5kids', 'cf5age', 'cm5age', 'cm5hhinc', 'cf5hhinc', 'cf5hhincb', 'cm5povco', 'cf5povco', 'cf5povcob']
motherAll = ['challengeID', 'cm1age', 'cm1adult', 'cm1kids', 'cm1hhinc', 'cm1inpov', 'cm2age', 'cm2amrf', 'cm2alvf', 'cm2fbir', 'cm2hhinc', 'cm2povco', 'cm3age', 'cm3amrf', 'cm3alvf', 'cm3adult', 'cm3kids', 'cm3hhinc', 'cm3povco', 'cm4age', 'cm4amrf', 'cm4alvf', 'cm4adult', 'cm4kids', 'cm4hhinc', 'cm4povco', 'cm5hhsize', 'cm5adult', 'cm5kids', 'cm5age', 'cm5hhinc', 'cm5povco']
fatherAll = ['challengeID', 'cf1age', 'cf1adult', 'cf1kids', 'cf1hhinc', 'cf1inpov', 'cf2age', 'cf2adult', 'cf2kids', 'cf2hhinc', 'cf2hhincb', 'cf2povco', 'cf2povcob', 'cf3age', 'cf3adult', 'cf3kids', 'cf3hhinc', 'cf3hhincb', 'cf3povco', 'cf3povcob', 'cf4age', 'cf4adult', 'cf4kids', 'cf4hhinc', 'cf4hhincb', 'cf4povco', 'cf4povcob', 'cf5hhsize', 'cf5adult', 'cf5kids', 'cf5age', 'cf5hhinc', 'cf5hhincb', 'cf5povco', 'cf5povcob']
motherFatherB = ['challengeID', 'cf1age', 'cf1adult', 'cf1kids', 'cf1hhinc', 'cf1inpov', 'cm1age', 'cm1adult', 'cm1kids', 'cm1hhinc', 'cm1inpov']
motherB = ['challengeID', 'cm1age', 'cm1adult', 'cm1kids', 'cm1hhinc', 'cm1inpov']
fatherB = ['challengeID', 'cf1age', 'cf1adult', 'cf1kids', 'cf1hhinc', 'cf1inpov']
motherFather1 = ['challengeID', 'cf2age', 'cf2adult', 'cf2kids', 'cf2hhinc', 'cf2hhincb', 'cf2povco', 'cf2povcob', 'cm2age', 'cm2amrf', 'cm2alvf', 'cm2fbir', 'cm2hhinc', 'cm2povco']
mother1 = ['challengeID', 'cm2age', 'cm2amrf', 'cm2alvf', 'cm2fbir', 'cm2hhinc', 'cm2povco']
father1 = ['challengeID', 'cf2age', 'cf2adult', 'cf2kids', 'cf2hhinc', 'cf2hhincb', 'cf2povco', 'cf2povcob']
motherFather3 = ['challengeID', 'cf3age', 'cf3adult', 'cf3kids', 'cf3hhinc', 'cf3hhincb', 'cf3povco', 'cf3povcob', 'cm3age', 'cm3amrf', 'cm3alvf', 'cm3adult', 'cm3kids', 'cm3hhinc', 'cm3povco']
mother3 = ['challengeID', 'cm3age', 'cm3amrf', 'cm3alvf', 'cm3adult', 'cm3kids', 'cm3hhinc', 'cm3povco']
father3 = ['challengeID', 'cf3age', 'cf3adult', 'cf3kids', 'cf3hhinc', 'cf3hhincb', 'cf3povco', 'cf3povcob']
motherFather5 = ['challengeID', 'cf4age', 'cf4adult', 'cf4kids', 'cf4hhinc', 'cf4hhincb', 'cf4povco', 'cf4povcob', 'cm4age', 'cm4amrf', 'cm4alvf', 'cm4adult', 'cm4kids', 'cm4hhinc', 'cm4povco']
mother5 = ['challengeID', 'cm4age', 'cm4amrf', 'cm4alvf', 'cm4adult', 'cm4kids', 'cm4hhinc', 'cm4povco']
father5 = ['challengeID', 'cf4age', 'cf4adult', 'cf4kids', 'cf4hhinc', 'cf4hhincb', 'cf4povco', 'cf4povcob']
motherFather9 = ['challengeID', 'cm5hhsize', 'cm5adult', 'cm5kids', 'cf5hhsize', 'cf5adult', 'cf5kids', 'cf5age', 'cm5age', 'cm5hhinc', 'cf5hhinc', 'cf5hhincb', 'cm5povco', 'cf5povco', 'cf5povcob']
mother9 = ['challengeID', 'cm5hhsize', 'cm5adult', 'cm5kids', 'cm5age', 'cm5hhinc', 'cm5povco']
father9 = ['challengeID', 'cf5hhsize', 'cf5adult', 'cf5kids', 'cf5age', 'cf5hhinc', 'cf5hhincb', 'cf5povco', 'cf5povcob']

# Load training data
# old = pd.read_csv(features)
# #GENERIC - CAN CHANGE OUT THE FEATURES FOR ANY APPROPRIATE BUNCHING. FOCUSING ON CONTINUOUS ONES FOR NOW
# #SHOULD EXPAND FROM CONTINUOUS TO OTHER TYPES
# new = old[motherFatherAll].copy()
#
# new.to_csv('motherFatherAll.csv', index=False)

def selectDF(featureNames):
    old = pd.read_csv(path.expanduser(r'C:\Users\Nick Kim\Documents\Nicholas Kim\School\School Semesters\Spring 2019 Semester\Fundamentals of Machine Learning\Assignment 2\Fragile Families\outputModified.csv'))
    new = old[motherFatherAll].copy()
    return new
# # Create word array
# with open("out_vocab_5.txt", 'r') as file:
#     for word in file:
#         word = word[:-1]
#         words.append(word)
#
# # Load testing data
# X_test = np.empty(shape = (600, 541))
# y_test = np.empty(shape = (600,))
#
# with open("test.txt", 'r') as file:
#     for i, line in enumerate(file):
#         y_test[i] = line[-2]
#         review = line[7:-4]
#         for word2 in review.split(' '):
#             for j, vocab in enumerate(words):
#                 if (preprocessSentences_v3.stem(word2).lower() == vocab):
#                     X_test[i][j] += 1
#
#
# model = GradientBoostingClassifier() #CHANGE THIS LINE TO TEST DIFFERENT MODELS
#
# X_new = np.append(X_train, X_test, axis = 0)
# y_new = np.append(y_train, y_test, axis = 0)
# skf = StratifiedKFold(n_splits=5)
# skf.get_n_splits(X_new, y_new)
#
# sum = 0
# # 5 fold cross validation
# for train_index, test_index in skf.split(X_new, y_new):
#     X_train, X_test = X_new[train_index], X_new[test_index]
#     y_train, y_test = y_new[train_index], y_new[test_index]
#     model.fit(X_train, y_train)
#     sum += (model.score(X_test, y_test))
# print(sum/5)
