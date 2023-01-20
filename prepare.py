import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os
from env import host, username, password

from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer

#----------------------------------------------------------    

def prep_zillow(df):

    df = df.dropna()

    df = df.drop(columns= ['parcelid', 'taxamount'])

    df = df.rename(columns= {'bedroomcnt': 'bedrooms',
                            'bathroomcnt': 'bathrooms',
                            'calculatedfinishedsquarefeet': 'sqft',
                            'taxvaluedollarcnt':'tax_value',
                            'regionidzip': 'zip_code',
                            'yearbuilt': 'year_built'
                             })
    return df

#----------------------------------------------------------    

def remove_outliers(df, k, col_list):
    ''' 
    This function takes in a dataframe, the threshold and a list of columns 
    and returns the dataframe with outliers removed
    '''   
    for col in col_list:

        q1, q3 = df[col].quantile([.25, .75])  # get quartiles
        
        iqr = q3 - q1   # calculate interquartile range
        
        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound

        # return dataframe without outliers
        
        df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
        
    return df
    
#----------------------------------------------------------

def subset_df(df, stratify=None, seed=42):
    '''
    This function takes in a DataFrame and splits it into train, validate, test subsets for our modeling phase. Does not take in a stratify option.
    '''
    train, val_test = train_test_split(df, train_size=.6, random_state=seed)
    validate, test = train_test_split(val_test, train_size=.5, random_state=seed)
    
    print(train.shape, validate.shape, test.shape)
    
    return train, validate, test

#----------------------------------------------------------

def scale_data(train, validate, test, 
               scaler, columns_to_scale,
               return_scaler=False):
    '''
    Scales the 3 data splits. 
    Takes in train, validate, and test data splits and returns their scaled counterparts.
    If return_scalar is True, the scaler object will be returned as well
    '''
    
    # make copies of our original data so nothing gets messed up
    train_scaled = train.copy()
    validate_scaled = validate.copy()
    test_scaled = test.copy()
    # make the scaler (unsure if redundant with addition I made)
    scaler = scaler
    # fit the scaler
    scaler.fit(train[columns_to_scale])
    # applying the scaler:
    train_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(train[columns_to_scale]),
                                                  columns=train[columns_to_scale].columns.values).set_index([train.index.values])
                                                  
    validate_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(validate[columns_to_scale]),
                                             columns=validate[columns_to_scale].columns.values).set_index([validate.index.values])
    
    test_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(test[columns_to_scale]),
                                                 columns=test[columns_to_scale].columns.values).set_index([test.index.values])
    
    if return_scaler:
        
        return scaler, train_scaled, validate_scaled, test_scaled
    
    else:
        
        return train_scaled, validate_scaled, test_scaled
    
#---------------------------------------------------------- 

def visualize_scaler(scaler, df, columns_to_scale, bins=10):
    '''
    This function takes in a specific scaler, dataframe, and returns two visuals of that data,
    one prior to scaling and one after scaling.
    '''
    
    fig, axs = plt.subplots(len(columns_to_scale), 2, figsize=(16,9))
    df_scaled = df.copy()
    df_scaled[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
    for (ax1, ax2), col in zip(axs, columns_to_scale):
        ax1.hist(df[col], bins=bins)
        ax1.set(title=f'{col} before scaling', xlabel=col, ylabel='count')
        ax2.hist(df_scaled[col], bins=bins)
        ax2.set(title=f'{col} after scaling with {scaler.__class__.__name__}', xlabel=col, ylabel='count')
    plt.tight_layout()

#----------------------------------------------------------

def vis_scaler_inverse(scaler, df, columns_to_scale, bins=10):
    '''
    This function takes in a specific scaler, dataframe, and returns two visuals of that data,
    one prior to scaling and one after scaling. Specifically for visualizations, doesn't return anything.
    '''
    
    fig, axs = plt.subplots(len(columns_to_scale), 3, figsize=(16,9))
    
    df_scaled = df.copy()
    df_scaled[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
    
    df_inverse = df.copy()
    df_inverse[columns_to_scale] = scaler.inverse_transform(df[columns_to_scale])

    for (ax1, ax2, ax3), col in zip(axs, columns_to_scale):
        
        ax1.hist(df[col], bins=bins)
        ax1.set(title=f'{col} before scaling', xlabel=col, ylabel='count')
        
        ax2.hist(df_scaled[col], bins=bins)
        ax2.set(title=f'{col} after scaling with {scaler.__class__.__name__}', xlabel=col, ylabel='count')
        
        ax3.hist(df_inverse[col], bins=bins)
        ax3.set(title=f'{col} after inverse transform {scaler.__class__.__name__}', xlabel=col, ylabel='count')
        
    plt.tight_layout()
    
#----------------------------------------------------------    

def compare_plots(transformed_data, train, target):
    '''
    This function will take in a train dataset and a transformed train dataset, bin them, and make a basic pairplot to visualize them side by side.
    '''
    
    plt.subplot(121)
    plt.hist(train[target], bins=25)
    plt.title('Original Data')
    plt.show()
    
    plt.subplot(122)
    plt.hist(transformed_data, bins=25)
    plt.title('Transformed Data')
    
#----------------------------------------------------------

def hist_charts(df):

    for col in df:

        plt.hist(df[col], bins=25)
        plt.title(f'{col} distribution')
        plt.show()
        
#----------------------------------------------------------
