import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('../data/azureml/Bike_Rental_UCI_dataset.csv')

def day_of_week():
    ## First day in the dataset is Saturday
    days = pd.DataFrame([[0, 1, 2, 3, 4, 5, 6],
      ["Sun", "Mon", "Tue", "Wed", "Thr", "Fri", "Sat"]]).transpose()
    days.columns = ['weekday', 'dayOfWeek']
    return days

days_df = day_of_week()
days_df.head()


df = pd.merge(df, days_df, on='weekday', how='outer')
df.head()


def set_days(df):
    number_of_rows = df.shape[0]
    df['days'] = pd.Series(range(number_of_rows))/24
    df['days'] = df['days'].astype('int')
    return df


set_days(df)

print("Done...")