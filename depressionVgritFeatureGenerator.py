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

# Reads from background.csv
features = path.expanduser(r'C:\Users\Nick Kim\Documents\Nicholas Kim\School\School Semesters\Spring 2019 Semester\Fundamentals of Machine Learning\Assignment 2\Fragile Families\background.csv')

#want to decrease loading times
#Write out a csv that only contains the columns from the relevant categories


# Load training data
old = pd.read_csv(features)
newold = old[['challengeID', 'cf2md_case_con', 'cf2md_case_lib', 'cm2md_case_con', 'cm2md_case_lib', 'cf3md_case_con', 'cf3md_case_lib', 'cm3md_case_con', 'cm3md_case_lib', 'cf4md_case_con', 'cf4md_case_lib', 'cm4md_case_con', 'cm4md_case_lib', 'cf5md_case_con', 'cf5md_case_lib', 'cm5md_case_con', 'cm5md_case_lib']].copy()
colnames = ['challengeID', 'cf2md_case_con', 'cf2md_case_lib', 'cm2md_case_con', 'cm2md_case_lib', 'cf3md_case_con', 'cf3md_case_lib', 'cm3md_case_con', 'cm3md_case_lib', 'cf4md_case_con', 'cf4md_case_lib', 'cm4md_case_con', 'cm4md_case_lib', 'cf5md_case_con', 'cf5md_case_lib', 'cm5md_case_con', 'cm5md_case_lib']

new = newold.astype(np.float64)
new['challengeID'] = new['challengeID'].astype(np.int)


for colName in colnames:
    zeroCount = 0.0
    oneCount = 0.0
    for i, val in enumerate(new[colName]):
        if val == 0:
            zeroCount += 1.0
        if val == 1:
            oneCount += 1.0
    print(colName)
    newVal = (oneCount)/(oneCount+zeroCount)
    print(newVal)
    for i, val in enumerate(new[colName]):
        if val < 0:
            new[colName][i] = newVal

new.to_csv('depressionFeatures.csv', index=False)

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
