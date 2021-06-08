import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import env
import os

def wrangle_telco():
    '''
    Read telco_data csv file into a pandas DataFrame,
    only returns two-year contract customers and desired columns,
    drop any rows with Null values, convert numeric columns to int or float,
    return cleaned telco DataFrame.
    '''
    # Acquire data from csv file.
    df = pd.read_csv("telco_data.csv")
    
    # Apply a mask for contract type
    df = df[df.contract_type == 'Two year']
    
    # Keep only the necessary columns
    df = df[['customer_id', 'tenure', 'monthly_charges', 'total_charges', 'churn']]
    
    # Replace white space values with NaN values.
    df = df.replace(r'^\s*$', np.nan, regex=True)
    
    # Drop all rows with NaN values.
    df = df.dropna()
    
    # Convert total charges column to float data type.
    df.total_charges = df.total_charges.astype('float64')
    
    return df

def wrangle_zillow():
    '''
    Read zillow csv file into a pandas DataFrame,
    only returns desired columns and single family residential properties,
    drop any rows with Null values, drop duplicates,
    return cleaned zillow DataFrame.
    '''
    # Acquire data from csv file.
    df = pd.read_csv('zillow.csv')
    
    # Drop nulls
    df = df.dropna()
    
    # Drop duplicates
    df = df.drop_duplicates()
    
    return df

def wrangle_mall():
    '''
    Checks for zillow.csv file and imports it if present. If absent, it will pull in bedroom bathroom counts, sq ft.
    tax value dollar count, year built, tax amount, and fips from properties 2017 in the zillow database. Then it will
    drop nulls and drop duplicates
    '''
    filename = 'mall.csv'
    if os.path.isfile(filename):
        mall_df = pd.read_csv(filename, index_col=0)
        return mall_df
    else:
        mall_df = pd.read_sql('''SELECT * FROM customers''', env.get_db_url('mall_customers'))
        mall_df = mall_df.dropna()
        mall_df = mall_df.drop_duplicates()
        mall_df.to_csv('mall.csv')
        return mall_df 