import pandas as pd
import numpy as np

def fillMissing(inputcsv, outputcsv):

    # read input csv - takes time
    df = pd.read_csv(inputcsv, low_memory=False)
    # Fix date bug
    #df.cf4fint = ((pd.to_datetime(df.cf4fint) - pd.to_datetime('1960-01-01')) / np.timedelta64(1, 'D')).astype(int)

    # num = df._get_numeric_data()
    # num[num < 0] = np.nan
    # print(num)
    df[df < 0] = np.nan
    # replace NA's with median
    df = df.fillna(df.median())
    # if still NA, replace with 1
    df = df.fillna(value=1)
    print(df)

    # write filled outputcsv
    df.to_csv(outputcsv, index=False)

# Usage:
fillMissing('background.csv', 'outputModified.csv')
filleddf = pd.read_csv('outputModified.csv', low_memory=False)
