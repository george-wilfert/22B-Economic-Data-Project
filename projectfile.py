import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn import linear_model
import mpl_toolkits.mplot3d
from datetime import *
from mpl_toolkits.mplot3d import Axes3D

#3d Model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import mpl_toolkits.mplot3d

#reading in data and metadata
data = pd.read_csv("data.csv")
meta_data = pd.read_csv("metadata.csv")

#getting Total Manufacturing in millions of dollars and cleaning df
mtm = data.loc[data['time_series_code'] == 'MTM_TI_US_adj']
print(mtm)
mtm = mtm.drop('time_series_code', axis=1)

mtm['value'] = mtm['value'].astype('int')
mtm.date = pd.to_datetime(mtm['date'])

#Creating lineplot
sns.lineplot(data=mtm, x="date", y="value")

#getting Balance of Payment Goods and Services Imports,  Millions of Dollars and cleaning df
imp = data.loc[data['time_series_code'] == 'BOPGS_IMP_US_adj']
imp = imp.drop('time_series_code', axis=1)

imp['value'] = imp['value'].astype('int')
imp.date = pd.to_datetime(imp['date'])

sns.lineplot(data=imp, x="date", y="value")

#changes value to numberic and coerces any NaN
data['value'] = pd.to_numeric(data['value'], errors="coerce")

#Key indicators. to filter from the metadata
keys = ["Housing Units Completed", 'New Single-family House Sold',
        'Housing Units Under Construction', 'Housing Units Started',
        'Housing Units Authorized But Not Started', 'Annual Rate for Housing Units Authorized in Permit-Issuing Places']

#combining data frames and making lineplot to print
list_of_df = [imp,mtm]
imported_and_manufacturing = pd.concat(list_of_df)
sns.lineplot(data=imported_and_manufacturing, x="date",y="value")


#changes value to numberic and coerces any NaN
data['value'] = pd.to_numeric(data['value'], errors="coerce")

#Key indicators to filter from the metadata
keys = ["Housing Units Completed", 'New Single-family House Sold',
        'Housing Units Under Construction', 'Housing Units Started',
        'Housing Units Authorized But Not Started', 'Annual Rate for Housing Units Authorized in Permit-Issuing Places']

#any() returns True or False checks if key word is in in the row and then creating a new df
filter_meta = filter(lambda x: any(key in x[1]['cat_desc'] for key in keys),meta_data.iterrows())
housing_meta = pd.DataFrame([item[1] for item in filter_meta])

#selects row from the metdata where 'cat_desc' had any of the key indicatoes
time_series_codes = housing_meta['time_series_code'].tolist()

#new dataframe for the housing indicators
housing_data = data[data['time_series_code'].isin(time_series_codes)]

#converst the 'date' column so we can ogranize it in chronological order
housing_data['date'] = pd.to_datetime(housing_data['date'])
housing_data.sort_values('date', inplace = True)

#function to create plots(repetative)
def pl_data(data, title, label, color = 'blue'):
    plt.figure(figsize = (12,6))
    sns.lineplot(data = data, x = 'date', y = 'value', label = label, color = color)
    plt.title(title)
    plt.xlabel('Year')
    plt.ylabel('Annual Rate')
    plt.legend()
    plt.show()

