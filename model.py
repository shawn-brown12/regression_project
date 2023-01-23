import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from math import sqrt
from scipy.stats import pearsonr, spearmanr

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix

from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, PolynomialFeatures
from sklearn.linear_model import LassoLars, LinearRegression, TweedieRegressor

import warnings
warnings.filterwarnings("ignore")

seed = 42
    
#---------------------------------------------------
    
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

#---------------------------------------------------

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
    
#---------------------------------------------------

def regression_errors(y, yhat):
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
    
    return mse, sse, rmse, ess, tss

#---------------------------------------------------

def baseline_mean_errors(y):
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

    return mse, sse, rmse

#---------------------------------------------------

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
        
#---------------------------------------------------

def rfe(n_features, X_train, y_train):
    '''
    This function will take in a number of features, an X_train, and a y_train, to determine which features (based on the number chosen) are best suited to modeling for the best RMSE based on the rfe method.
    '''
    lm = LinearRegression()
    rfe = RFE(lm, n_features_to_select=n_features)

    rfe.fit(X_train, y_train)

    ranks = rfe.ranking_
    columns = X_train.columns.tolist()
    
    feature_ranks = pd.DataFrame({'ranking': ranks,
                              'feature': columns})

    feature_ranks = feature_ranks.sort_values('ranking')

    return pd.DataFrame(feature_ranks).head(n_features)

#---------------------------------------------------

def f_selector(k, X_train, y_train):
    '''
    This function will take in a number, an X_train, and a y_train, to determine which features (based on the number chosen) are best suited to modeling for the best RMSE, based upon the SelectKBest method.
    '''
    f_selector = SelectKBest(f_regression, k=k)

    f_selector.fit(X_train, y_train)

    f_select_mask = f_selector.get_support()

    select_k_best_features = X_train.iloc[:,f_select_mask]
    
    #print(select_k_best_features.head(k))
    
    return pd.DataFrame(select_k_best_features)

#---------------------------------------------------

def create_preds_df(y_train):
    '''
    This function will create a dataframe, 'preds_df', using the y_train (or validate or test), and create a baseline column (using median), and return the dataframe.
    '''
    preds_df = pd.DataFrame({'actual': y_train})

    preds_df['baseline_median'] = y_train.median()
    
    return preds_df

#---------------------------------------------------

def lin_regression(X_train, y_train, preds_df):
    '''
    This function will take in an X_train, y_train, and a preds_df (or validate or test), and create, fit and predict upon a simple Linearregression() model, add the predictions to the preds_df, and return it.
    '''
    lm = LinearRegression()

    lm.fit(X_train, y_train)
    
    preds_df['lm_preds'] = lm.predict(X_train)
    
    return preds_df

#---------------------------------------------------

def lasso_lars(X_train, y_train, preds_df, alpha=.1):
    '''
    This function will take in an X_train, y_train, and a preds_df (or validate or test) and an alpha (with a default of 0.1), and create, fit and predict upon a simple LassoLars() model, add the predictions to the preds_df, and return it. 
    '''
    lasso = LassoLars(alpha=alpha)

    lasso.fit(X_train, y_train)

    preds_df['lasso_preds'] = lasso.predict(X_train)
    
    return preds_df

#---------------------------------------------------

def glm_model(X_train, y_train, preds_df, power=0):
    '''
    This function will take in an X_train, y_train, a preds_df, and a power (with a default of 0), and create, fit, and predict upon the TweedieRegressor() (GLM) model, add those predictions to a preds_df, and return it.
    '''
    tweedie = TweedieRegressor(power=power)

    tweedie.fit(X_train, y_train)

    preds_df['tweedie_preds'] = tweedie.predict(X_train)
    
    return preds_df

#---------------------------------------------------

def poly_subset(X_train, y_train, degree=2):
    '''
    This function will take in an X_train, a y_train, as well as a number for degrees chosen (defaulting to 2), to create a polynomial subset of the data, fitting and transforming it for use in other regression models and returning that new subset.
    '''
    pf = PolynomialFeatures(degree=degree)

    pf.fit(X_train, y_train)

    X_polynomial = pf.transform(X_train)
    
    return X_polynomial

#---------------------------------------------------

def poly_model(X_polynomial, y_train, preds_df, m):
    '''
    This function will take in a polynomial subset, a y_train, the preds_df, and a model type (m), to create, fit, and predict on the model, as well as creating a column for it in the preds_df, and then returning it.
    '''
    model = m

    model.fit(X_polynomial, y_train)

    preds_df['poly_preds'] = model.predict(X_polynomial)
    
    return preds_df

#---------------------------------------------------

def get_rmses(preds_df):
    '''
    This function will take in the preds_df, as long as there's a column for each of four different algorithms in the preds_df, and using those, create another dataframe with the rmse values for each algorithm and return it.
    '''
    lm_rmse = sqrt(mean_squared_error(preds_df['lm_preds'], preds_df['actual']))
    lasso_rmse = sqrt(mean_squared_error(preds_df['actual'], preds_df['lasso_preds']))
    tweedie_rmse = sqrt(mean_squared_error(preds_df['actual'], preds_df['tweedie_preds']))
    poly_rmse = sqrt(mean_squared_error(preds_df['actual'], preds_df['poly_preds']))
    
    results = pd.DataFrame({'model':['linear', 'lasso', 'tweedie_norm', 'linear_poly'],
                            'rmse':[lm_rmse, lasso_rmse, tweedie_rmse, poly_rmse]})

    return results
    
#---------------------------------------------------

def viz_6(final_stats):
    '''
    This is a function for the sixth visual in the final report
    '''
    sns.barplot(data=final_stats, x='model', y='results')

    plt.title('Test Results compared to Baseline')
    plt.xlabel('Model Name')
    plt.ylabel('Number (on Average) off of Actual Home Value')

    plt.show()
    
#---------------------------------------------------

