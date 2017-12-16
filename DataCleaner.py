# coding: utf-8
# # Quant Trading
# ### Import Libraries
#Dataset: https://www.kaggle.com/benjibb/lstm-stock-prediction-20170507
#Cleaning Guide: https://app.pluralsight.com/library/courses/python-understanding-machine-learning/table-of-contents

import pandas as pd                 #Dataframe library
import matplotlib.pyplot as plt     #Plots data
import numpy as np                  #Provides N-dim object support
from sklearn import preprocessing   #For normalization

class DataCleaner():
    def __init__(self):
        return
        
    def get_clean_data(self, training_run, test=False):
        #do ploting inline instead of in a seperate window
        #get_ipython().magic('matplotlib inline')

        # # 1) Load and review data
        df = pd.read_csv("./data/prices-split-adjusted.csv")
        df.head(5)                          #Display first 5 data entries

        #df.shape            #shows rows x cols

        # ### Isolate Desired Stock
        df = df[df.symbol == 'NFLX']
        #df = df[df.symbol == 'GOOGL']
        del df['volume']
        del df['symbol']
        del df['date']
        df.head(5)
        df.shape
        #1762 days of data
        #Train = 1412              80.13%
        #Test = 350                19.86%
        #4Fold = 353               Validation matches test


        # # 2) Split Data via 4FoldCV to train test and validation
        X_train = df[0:1412]
        X_test = df[1413:1761]

        #Training Data 4 Fold
        X_train1 =X_train[353:1412]
        X_train2 =np.vstack((X_train[0:352], X_train[706:1412]))
        X_train3 =np.vstack((X_train[0:705], X_train[1059:1412]))
        X_train4 =X_train[0:1058]

        #Validation Data
        X_Val1 = df[0:352]
        X_Val2 = df[353:705]
        X_Val3 = df[706:1058]
        X_Val4 = df[1059:1412]


        # # 4) Normalization
        #Use normalization in problems where data is sensitive to magnitude
        #Split then normalize to keep everything independent

        X1_train_norm = preprocessing.normalize(X_train1, norm='l2')
        X2_train_norm = preprocessing.normalize(X_train2, norm='l2')
        X3_train_norm = preprocessing.normalize(X_train3, norm='l2')
        X4_train_norm = preprocessing.normalize(X_train4, norm='l2')

        X1_val_norm = preprocessing.normalize(X_Val1, norm='l2')
        X2_val_norm = preprocessing.normalize(X_Val2, norm='l2')
        X3_val_norm = preprocessing.normalize(X_Val3, norm='l2')
        X4_val_norm = preprocessing.normalize(X_Val4, norm='l2')

        X_test_norm = preprocessing.normalize(X_test, norm='l2')

        if test:
            return X_test_norm
        
        if training_run == 1:
            return [X1_train_norm,X1_val_norm]
        if training_run == 2:
            return [X2_train_norm,X2_val_norm]
        if training_run == 3:
            return [X3_train_norm,X3_val_norm]
        if training_run == 4:
            return [X4_train_norm,X4_val_norm]