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
    
    f_selector = SelectKBest(f_regression, k=k)

    f_selector.fit(X_train, y_train)

    f_select_mask = f_selector.get_support()

    select_k_best_features = X_train.iloc[:,f_select_mask]
    
    #print(select_k_best_features.head(k))
    
    return pd.DataFrame(select_k_best_features)

#---------------------------------------------------

def create_preds_df(y_train):
    
    preds_df = pd.DataFrame({'actual': y_train})

    preds_df['baseline_median'] = y_train.median()
    
    return preds_df

#---------------------------------------------------

def lin_regression(X_train, y_train):

    lm = LinearRegression()

    lm.fit(X_train, y_train)
    
    preds_df['lm_preds'] = lm.predict(X_train)
    
    return preds_df

#---------------------------------------------------

def lasso_lars(X_train, y_train, alpha=.1):
    
    lasso = LassoLars(alpha=alpha)

    lasso.fit(X_train, y_train)

    preds_df['lasso_preds'] = lasso.predict(X_train)
    
    return preds_df

#---------------------------------------------------

def glm_model(X_train, y_train, power=0):
    
    tweedie = TweedieRegressor(power=power)

    tweedie.fit(X_train, y_train)

    preds_df['tweedie_preds'] = tweedie.predict(X_train)
    
    return preds_df

#---------------------------------------------------

def poly_subset(X_train, y_train, degree=2):
    
    pf = PolynomialFeatures(degree=degree)

    pf.fit(X_train, y_train)

    X_polynomial = pf.transform(X_train)
    
    return X_polynomial

#---------------------------------------------------

def poly_model(X_polynomial, y_train, m):
    
    model = m

    model.fit(X_polynomial, y_train)

    preds_df['poly_preds'] = model.predict(X_polynomial)
    
    return preds_df

#---------------------------------------------------

def get_rmses(preds_df):
    
    lm_rmse = sqrt(mean_squared_error(preds_df['lm_preds'], preds_df['actual']))
    lasso_rmse = sqrt(mean_squared_error(preds_df['actual'], preds_df['lasso_preds']))
    tweedie_rmse = sqrt(mean_squared_error(preds_df['actual'], preds_df['tweedie_preds']))
    poly_rmse = sqrt(mean_squared_error(preds_df['actual'], preds_df['poly_preds']))
    
    print(f'Linear Regression RMSE is: {lm_rmse}')
    print(f'Lasso-Lars Regression RMSE is: {lasso_rmse}')
    print(f'GLM Regression RMSE is: {tweedie_rmse}')
    print(f'Polynomial Regression RMSE is: {poly_rmse}')
    
    results = pd.DataFrame({'model':['linear', 'lasso', 'tweedie_norm', 'linear_poly'],
                            'rmse':[lm_rmse, lasso_rmse, tweedie_norm, poly_rmse]})

    return results
    
#---------------------------------------------------


#---------------------------------------------------

