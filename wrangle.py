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

def get_connection(db, user=username, host=host, password=password):
    '''
    This functions imports my credentials for the Codeup MySQL server to be used to pull data
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

#----------------------------------------------------------    
    
#simply copied this for the framework, will be changed for other data
def get_zillow():
    '''
    This function will check locally if there's a zillow.csv file in the local directory, and if not, working with the 
    get_connection function, will pull the zillow dataset from the Codeup MySQL server. After that, it will also save a copy of 
    the csv locally if there wasn't one, so it doesn't have to run the query each time. Additionally, this will clean up by dataset to be ready to be split and used for modeling.
    '''
    if os.path.isfile('zillow_2017.csv'):
        
        df = pd.read_csv('zillow_2017.csv')
        df = df.drop(columns='Unnamed: 0')

        return df
    
    else:
        
        url = get_connection('zillow')
        query = '''
                SELECT bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, 
                       taxvaluedollarcnt, yearbuilt, taxamount, fips
                FROM properties_2017
                JOIN propertylandusetype USING(propertylandusetypeid)
                WHERE propertylandusedesc = 'Single Family Residential';
                '''
        df = pd.read_sql(query, url)        
        df = df.dropna()

        df.bedroomcnt = df.bedroomcnt.astype(int)
        df.calculatedfinishedsquarefeet = df.calculatedfinishedsquarefeet.astype(int)
        df.taxvaluedollarcnt = df.taxvaluedollarcnt.astype(int)
        df.yearbuilt = df.yearbuilt.astype(int)
        df.fips = df.fips.astype(int)
    
        df = df.rename(columns= {'bedroomcnt': 'bedrooms',
                                 'bathroomcnt': 'bathrooms',
                                 'calculatedfinishedsquarefeet': 'square_ft',
                                 'taxvaluedollarcnt':'tax_value',
                                 'yearbuilt':'built',
                                 'taxamaount':'taxes'
                                 })        
        df.to_csv('zillow_2017.csv')

        return df

#----------------------------------------------------------            
    
def get_auto_mpg():
    '''
    Acquire, clean, and return the auto-mpg dataset
    '''
    
    df = pd.read_fwf('auto-mpg.data', header=None)
    
    df.columns = ['mpg', 'cylinders', 'displ', 'horsepower', 'weight', 'acc',
                  'model_year', 'origin', 'name']
    
    df = df[df['horsepower'] != '?']
    
    df['horsepower'] = df['horsepower'].astype('float')
    
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
    
    return train, validate, test

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

def chi2_report(df, col, target):
    '''
    This function is to be used to generate a crosstab for my observed data, and use that the run a chi2 test, and generate the report values from the test.
    '''
    
    observed = pd.crosstab(df[col], df[target])
    
    chi2, p, degf, expected = stats.chi2_contingency(observed)

    alpha = .05
    seed = 42
    
    print('Observed Values\n')
    print(observed.values)
    
    print('---\nExpected Values\n')
    print(expected.astype(int))
    print('---\n')

    print(f'chi^2 = {chi2:.4f}') 
    print(f'p     = {p:.4f}')

    print('Is p-value < alpha?', p < alpha)