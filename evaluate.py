import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

import env
import wrangle, prepare, explore

import math

def plot_residuals(y, yhat, df):
    '''
    This function takes in a y value and predicted value and outputs a
    residual plot.
    Assumes y and yhat are columns from a Pandas DataFrame.
    '''
    df.residuals = df[y] - df[yhat]
    df.residuals.plot.hist()

def regression_errors(y, yhat, df):
    '''
    This function takes in a y value and predicted value.
    Outputs the following:
    Sum of squared errors
    Explained sum of squares
    Total sum of squares
    Mean squared error
    Root mean squared error
    '''
    df.residuals = df[y] - df[yhat]
    print(f'Sum of squared errors (SSE): {(df.residuals ** 2).sum()}')
    print(f'Explained sum of squares (ESS): {((df[yhat] - df[y].mean())**2).sum()}')
    print(f'Total sum of squares (TSS): {((df[y] - df[y].mean())**2).sum()}')
    print(f'Mean squared error (MSE): {mean_squared_error(df[y], df[yhat])}')
    print(f'Root mean squared error (RMSE): {math.sqrt(mean_squared_error(df[y], df[yhat]))}')

def baseline_mean_errors(y, df):
    '''
    This function takes in a y value and a dataframe and returns SSE, MSE, and RMSE
    for the baseline model.
    '''
    df['yhat_baseline'] = df[y].mean()
    df.baseline_residuals = df[y] - df.yhat_baseline
    sse_baseline = (df.baseline_residuals ** 2).sum()
    mse_baseline = sse_baseline / df.shape[0]
    rmse_baseline = math.sqrt(mse_baseline)

    print(f'''
    Baseline:

    sse:  {sse_baseline}
    mse:  {mse_baseline}
    rmse: {rmse_baseline}
    ''')

def better_than_baseline(y, yhat, df):
    '''
    Takes in y, yhat, and dataframe and returns whether the model performs better than the baseline
    '''
    df['yhat_baseline'] = df[y].mean()
    df.baseline_residuals = df[y] - df.yhat_baseline
    sse_baseline = (df.baseline_residuals ** 2).sum()
    mse_baseline = sse_baseline / df.shape[0]
    rmse_baseline = math.sqrt(mse_baseline)

    rmse = math.sqrt(mean_squared_error(df[y], df[yhat]))
    return rmse < rmse_baseline