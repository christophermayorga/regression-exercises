# Regression Exercises
![Relevant xkcd](https://imgs.xkcd.com/comics/curve_fitting.png)
> Cauchy-Lorentz: "Something alarmingly mathematical is happening, and you should probably pause to Google my name and check what field I originally worked in."

## Wrangle Exercises
- Purpose was to write functions to imput and clean data so that it can be used in a regression model
- Wrote a function to wrangle telco data and zillow data
- I want to rewrite the wrangle functions to instead be more generalizable and useable on any database.
    - Will need to rewrite the write function to take in db_name and add .csv to the end as a filename. Then I should only return df rather than zillow_df for instance. 
    - Will also have to figure out a way to give more flexibility with my cleaning actions, which may need to be done with passing way more arguments. 

## Scaling Exercises
- Purpose of these exercises are to develop a function to scale numerical data appropriately for machine learning
- Wrote three functions for different scaling mechanisms, min max, standard scaler, robust scaling, and quantile transformer. 
    - The functions take in a data set and scales the numerical values within according to the scaler. It then puts it into a dataframe and renames the columns with the column names of the data set passed into it. As is, this function is where I want it. I could consolidate all three into one using if/elif statements, but it does seem like it would be making it nedlessly complicated when it already works fine. 
    