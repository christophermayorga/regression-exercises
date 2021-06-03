import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import env

def wrangle_telco():
    '''
    Read telco_data csv file into a pandas DataFrame,
    only returns two contract customers and desired columns,
    drop any rows with Null values, convert numeric columns to int or float,
    return cleaned telco DataFrame.
    '''
    # Acquire data from csv file.
    df = pd.read_csv("telco_data.csv")
    
    # Apply a mask for contract type
    df = df[df.contract_type == 'Two year']
    
    # Keep only the necessary columns
    df = df[['customer_id', 'tenure', 'monthly_charges', 'total_charges']]
    
    # Replace white space values with NaN values.
    df = df.replace(r'^\s*$', np.nan, regex=True)
    
    # Drop all rows with NaN values.
    df = df.dropna()
    
    # Convert total charges column to float data type.
    df.total_charges = df.total_charges.astype('float64')
    
    return df

