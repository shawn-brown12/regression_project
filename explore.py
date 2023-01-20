import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from pydataset import data

import matplotlib.pyplot as plt
import seaborn as sns

import math
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import explained_variance_score

#---------------------------------------------------

def chi2_report(df, col, target):
    '''
    This function is to be used to generate a crosstab for my observed data, and use that the run a chi2 test, and generate the report values from the test.
    '''
    
    alpha = .05
    seed = 42
    
    observed = pd.crosstab(df[col], df[target])
    
    chi2, p, degf, expected = stats.chi2_contingency(observed)
    
    print('Observed Values\n')
    print(observed.values)
    
    print('---\nExpected Values\n')
    print(expected.astype(int))
    print('---\n')

    print(f'chi^2 = {chi2:.4f}') 
    print(f'p     = {p:.4f}')

    print('Is p-value < alpha?', p < alpha)

#---------------------------------------------------

def ind_ttest_report(group1, group2):
    '''
  
    '''
    
    t, p = stats.ttest_ind(group1, group2, equal_var=False)

    alpha = .05
    seed = 42

    print(f'T-statistic = {t:.4f}') 
    print(f'p-value     = {p:.4f}')

    print('Is p-value < alpha?', p < alpha)

#---------------------------------------------------
    
def mann_whitney_report(group1, group2):
    '''
  
    '''
    
    statistic, p = stats.mannwhitneyu(group1, group2)

    alpha = .05
    seed = 42

    print(f'T-statistic = {statistic:.4f}') 
    print(f'p-value     = {p:.4f}')

    print('Is p-value < alpha?', p < alpha)

#---------------------------------------------------

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

#---------------------------------------------------

def first_viz(train):

    plt.figure(figsize=(7,6))
    sns.barplot(x='bedrooms', y='bathrooms', data=train, palette='cool')
    plt.title('Do Bathrooms Increase as Bedrooms Do?')
    plt.xlabel('Number of Bedrooms')
    plt.ylabel('Number of Bathrooms')
    plt.show()