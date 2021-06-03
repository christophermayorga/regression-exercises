import pandas as pd
from env import user, password, host, get_db_url
import os
# must have env.py saved in same directory as script. ensure the env.py is in your .gitignore

def get_zillow_db(db_name = 'zillow', user=user, password=password, host=host):
    '''
    Imports single residential family properties from the zillow database. columns are bedroom/bathroom counts,
    square footage, tax value, year it was built, tax, and fips for the year 2017'''
    filename = 'zillow.csv'
    if os.path.isfile(filename):
        zillow_df = pd.read_csv(filename, index_col=0)
        return zillow_df
    else:
        zillow_df = pd.read_sql('''SELECT bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, 
                                          taxvaluedollarcnt, yearbuilt, taxamount, fips 
                                          FROM properties_2017
                                          WHERE propertylandusetypeid = 261;''',
                        get_db_url('zillow'))
        zillow_df.to_csv(filename)
        return zillow_df