# Prepare the telco dataset before beginning exploration.
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
import sklearn.feature_selection
from sklearn.linear_model import LinearRegression

# turn off warnings
import warnings
warnings.filterwarnings("ignore")

# The prep function returns the train, validate and test splits:
def prep_telco_data(df):
    '''
    Splits telco df into train, validate, test. Best if use wrangle_telco()
    from wrangle.py as the input.
    '''
    # All the cleaning has been done in wrangle
    # Now just split
    train, test = train_test_split(df, train_size=0.8, random_state=1349)
    train, validate = train_test_split(train, train_size=0.7, random_state=1349)
    return train, validate, test

def scale_telco_data(train, validate, test):
    '''
    This function takes in train, validate, and test dateframes and returns
    scaled versions of each dataframe. 
    '''
    # Use MinMaxScaler

    # create our scaler
    scaler_tenure = MinMaxScaler()
    scaler_monthlycharges = MinMaxScaler()
    scaler_totalcharges = MinMaxScaler()

    # fit our scaler
    scaler_tenure.fit(train[['tenure']])
    scaler_monthlycharges.fit(train[['monthly_charges']])
    scaler_totalcharges.fit(train[['total_charges']])

    # use our scaler
    train['tenure'] = scaler_tenure.transform(train[['tenure']])
    train['monthly_charges'] = scaler_monthlycharges.transform(train[['monthly_charges']])
    train['total_charges'] = scaler_totalcharges.transform(train[['total_charges']])

    validate['tenure'] = scaler_tenure.transform(validate[['tenure']])
    validate['monthly_charges'] = scaler_monthlycharges.transform(validate[['monthly_charges']])
    validate['total_charges'] = scaler_totalcharges.transform(validate[['total_charges']])
    
    test['tenure'] = scaler_tenure.transform(test[['tenure']])
    test['monthly_charges'] = scaler_monthlycharges.transform(test[['monthly_charges']])
    test['total_charges'] = scaler_totalcharges.transform(test[['total_charges']])

    return train, validate, test

def train_validate_test_split(df):
    '''
    Generic function. Takes in any dataframe and returns train, validate, and test splits.
    '''
    # Assumes data is clean
    # Now just split
    train, test = train_test_split(df, train_size=0.8, random_state=1349)
    train, validate = train_test_split(train, train_size=0.7, random_state=1349)
    return train, validate, test

def select_kbest(X, y, k):
    # make the object
    kbest = sklearn.feature_selection.SelectKBest(sklearn.feature_selection.f_regression, k=k)

    # fit the object
    kbest.fit(X, y)
    
    # use the object (.get_support() is that array of booleans to filter the list of column names)
    return X.columns[kbest.get_support()].tolist()

def select_rfe(X, y, k, return_rankings=False, model=LinearRegression()):
    # Use the passed model, LinearRegression by default
    rfe = sklearn.feature_selection.RFE(model, n_features_to_select=k)
    rfe.fit(X, y)
    features = X.columns[rfe.support_].tolist()
    if return_rankings:
        rankings = pd.Series(dict(zip(X.columns, rfe.ranking_)))
        return features, rankings
    else:
        return features