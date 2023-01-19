import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from pydataset import data

import matplotlib.pyplot as plt
import seaborn as sns

import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import explained_variance_score

#-------------------------------------------

def plot_residuals(y, yhat):
    '''
    This function takes in the actual value and predicted value 
    then creates a scatter plot of those values
    '''
    
    residuals = y - yhat
    
    plt.scatter(x=y, y=residuals)
    plt.xlabel('Home Value')
    plt.ylabel('Residuals')
    plt.title('Residual vs Home Value Plot')
    plt.show()
    
#-------------------------------------------

def regression_errors(y, yhat):
    '''
    This function takes in the actual value and predicted value 
    then outputs: the sse, ess, tss, mse, and rmse
    '''
    
    mse = mean_squared_error(y, yhat)
    sse = mse * len(y)
    rmse = math.sqrt(mse)
    ess = ((yhat - y.mean())**2).sum()
    tss = ess + sse
    
    return mse, sse, rmse, ess, tss

#-------------------------------------------

def regression_errors_print(y, yhat):
    '''
    This function takes in the actual value and predicted value 
    then outputs a print statement of sse, ess, tss, mse, and rmse
    '''
    mse = mean_squared_error(y, yhat)
    sse = mse * len(y)
    rmse = math.sqrt(mse)
    ess = ((yhat - y.mean())**2).sum()
    tss = ess + sse
        
    print(f''' 
            SSE: {sse: .4f}
            ESS: {ess: .4f}
            TSS: {tss: .4f}
            MSE: {mse: .4f}
            RMSE: {rmse: .4f}
            ''')
    
#-------------------------------------------

def baseline_mean_errors(y):
    '''
    This function takes in the actual values and outputs the mse, sse, and rmse
    '''
    baseline = np.repeat(y.mean(), len(y))
    
    mse = mean_squared_error(y, baseline)
    sse = mse * len(y)
    rmse = mse ** .5
    
    return mse, sse, rmse

#-------------------------------------------

def baseline_mean_errors_print(y):
    '''
    This function takes in the actual value and predicted value
    then outputs a print statement of the SSE, MSE, and RMSE for the baseline
    '''
    
    baseline = np.repeat(y.mean(), len(y))
    
    mse = mean_squared_error(y, baseline)
    sse = mse * len(y)
    rmse = mse ** .5
    
    print(f'''
            sse_baseline: {sse: .4f}
            mse_baseline: {mse: .4f}
            rmse_baseline: {rmse: .4f}
            ''')

#-------------------------------------------

def better_than_baseline(y, yhat):
    '''
    This function takes in the target and the prediction
    then returns a print statement 
    to inform us if the model outperforms the baseline
    '''
    
    sse, ess, tss, mse, rmse = regression_errors(y, yhat)
    
    sse_baseline, mse_baseline, rmse_baseline = baseline_mean_errors(y)
    
    if sse < sse_baseline:
        
        print('My OSL model performs better than baseline')
        
    else:
        
        print('My OSL model performs worse than baseline. :( )')
        
#-------------------------------------------

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

#-------------------------------------------       
    
#my_range=range(1,len(train.index) + 1)
#plt.figure(figsize=(4,10))
#plt.scatter(train['tax_value'], my_range, color='green', alpha=0.8 , label='Property Value')
#plt.axvline(.75, c='tomato')

#plt.legend()

#plt.yticks(my_range, train.index)
#plt.title("Drivers of Price", loc='center')
#plt.xlabel('Price-Low     Average = 0      Price-High')
#plt.ylabel('Feature')

#plt.show()