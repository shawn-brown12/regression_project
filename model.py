import os
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix

def calc_baseline(df, baseline, col):
    '''
    This function is used to create a column within the dataframe to make a baseline column, and then calculate the baseline accuracy. Needs to be optimized more, but functions as is currently. Make sure to use the word 'baseline' when calling function.
    '''
    
    seed = 42
    
    df[baseline] = df[col].value_counts().idxmax()    

    base = (df[col] == df[baseline]).mean()
    
    print(f'Baseline Accuracy is: {base:.3}')
    
def xy_subsets(train, validate, test, target):
    '''
    This function will separate each of my subsets for the dataset (train, validate, and test) and split them further into my x and y subsets for modeling. When running this, be sure to assign each of the six variables in the proper order, otherwise it will almost certainly mess up. (X_train, y_train, X_validate, y_validate, X_test, y_test).
    '''  
    seed = 42
    
    X_train = train.drop(columns=[target])
    y_train = train[target]

    X_validate = validate.drop(columns=[target])
    y_validate = validate[target]

    X_test = test.drop(columns=[target])
    y_test = test[target]
    
    return X_train, y_train, X_validate, y_validate, X_test, y_test

def train_vailidate_test_split(df, target, stratify=None):
    '''
     This function will take my dataset, turn it into train, validate, and test, and then further split them into the x and y subsets. When running this, be sure to assign each of the variables in the proper order, otherwise it will almost certainly mess up. (X_train, y_train, X_validate, y_validate, X_test, y_test)
    '''
    train_validate, test = train_test_split(df, train_size =.60, random_state = 42, stratify = df[target])
    train, validate = train_test_split(train_validate, train_size = .7, random_state = 42, stratify = train_validate[target])
    
    X_train = train.drop(columns=target)
    y_train = train[target]
    
    X_val = validate.drop(columns=target)
    y_val = validate[target]
    
    X_test = test.drop(columns=target)
    y_test = test[target]
    
    return train, validate, test, X_train, y_train, X_val, y_val, X_test, y_test