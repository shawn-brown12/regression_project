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
    This function takes in two groups (columns), and will perform an independent t-test on them and print out the t-statistic and p-value, as well as determine if the p-value is lower than a predetermined (.05) alpha
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
    This function takes in two groups (columns)< and will perform a MannWhitneyU statistical test on them, to compare the means of each group to determine if there is a relationship between them.
    '''
    statistic, p = stats.mannwhitneyu(group1, group2)

    alpha = .05
    seed = 42

    print(f'Statistic = {statistic:.4f}') 
    print(f'p-value     = {p:.4f}')
    print('Is p-value < alpha?', p < alpha)

#---------------------------------------------------

def pearsonr_test(group1, group2):
    '''
    This function will take in two groups (columns), and perform a pearsonr statistical test on them, to determine if there is linear correlation between the two.
    '''
    statistic, p = stats.pearsonr(group1, group2)
    
    alpha = .05
    seed = 42

    print(f'Statistic = {statistic:.4f}') 
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

def viz_1(train):
    '''
    This function will create the first visualization used in the final report.
    '''    
    sns.set_style("whitegrid")
    sns.lineplot(data=train, x='bedrooms', y='bathrooms')

    plt.title('Bedrooms and Bathrooms')
    plt.xlabel('Number of Bedrooms')
    plt.ylabel('Number of Bathrooms')

    plt.show()
    
#---------------------------------------------------

def viz_2(train):
    '''
    This function will create the second visualization used in the final report.
    '''
    sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
    sns.lineplot(data=train, x='year_built', y='tax_value')
    
    plt.title('Home Value as Time Advances')
    plt.xlabel('Year Built')
    plt.ylabel('Value')
    
    plt.show()
    
#---------------------------------------------------

def viz_3(train):
    '''
    This function will create the third visualization used in the final report.
    '''
    sns.set_style("whitegrid")
    sns.regplot(x='year_built', y='sqft', data=train.sample(2000), line_kws={'color':'firebrick'})
    
    plt.title('Year Built Compared to Size')
    plt.xlabel('Year Built')
    plt.ylabel('Square Feet of House')
    
    plt.show()
    
#---------------------------------------------------

def viz_4(train):
    '''
    This function will create the fourth visualization used in the final report.
    '''
    sns.set_style("whitegrid")
    sns.catplot(data=train, x="fips", y="sqft", kind="bar", height=6, aspect=.5)

    plt.title('Does Fips code Effect the Size of a House?')
    plt.xlabel('Fips Code')
    plt.ylabel('Size of Home (In square feet)')

    plt.show()

#---------------------------------------------------

def viz_5(train):
    '''
    This function will create the fifth visualization used in the final report.
    '''
    fig, ax = plt.subplots(figsize=(7,6))
    sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})

    bplot = sns.countplot(x='bathrooms', hue='fips', data=train)
    ax.bar_label(bplot.containers[0], padding= 6)
        
    plt.title('Does Fips matter when it comes to Bathroom Amount?')
    plt.xlabel('Number of Bathrooms')
    plt.ylabel('Number of Houses')
    plt.show() 
    
          
#---------------------------------------------------
    
    
#---------------------------------------------------
    
    
    