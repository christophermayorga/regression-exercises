import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr, spearmanr

import env
import wrangle
import prepare

def plot_univar(df, cols):
    '''
    This function takes in a dataframe and a list of columns within that
    dataframe.
    List of columns: each column should be wrapped in a string and should
    represent a numeric independent variable.
    The ouput is histograms for each variable as subplots.
    '''
    plt.figure(figsize=(16, 3))

    for i, col in enumerate(cols):
    
        # i starts at 0, but plot nos should start at 1
        plot_number = i + 1 
    
        # Create subplot.
        plt.subplot(1,len(cols), plot_number)
    
        # Title with column name.
        plt.title(col)
    
        # Display histogram for column.
        df[col].hist(bins=5)
    
        # Hide gridlines.
        plt.grid(False)
        
def plot_variable_pairs(data_set, hue):
    sns.pairplot(data_set, hue=hue)
    
def months_to_years(data_set):
    data_set['tenure_years'] = round(data_set.tenure / 12, 0)
    data_set = data_set.rename(columns={'tenure': 'tenure_month'})
    return data_set

def plot_categorical_and_continuous_vars(data_set, cat_var, con_var):
    sns.barplot(data=data_set, y=con_var, x=cat_var)
    plt.show()
    sns.violinplot(data=data_set, y=con_var, x=cat_var)
    plt.show()
    sns.boxplot(data=data_set, y=con_var, x=cat_var)
    plt.show()
    
def heat_corr(data_set):
    return sns.heatmap(data_set.corr(), cmap='Greens', annot=True, linewidth=0.5, mask= np.triu(data_set.corr()))