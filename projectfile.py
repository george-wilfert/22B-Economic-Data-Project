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

#plot for housing units started
starts = housing_data[housing_data['time_series_code'].str.contains('ASTARTS_TOTAL_US')]
pl_data(starts, 'Annual Rate for Housing Units Started', 'National Starts')

#Regonal data for housing using started
regions = ['NE', 'MW', 'SO', 'WE']
for region in regions:
    region_data = housing_data[housing_data['time_series_code'].str.contains(f'ASTARTS_TOTAL_{region}_adj')]
    sns.lineplot(data = region_data, x = 'date', y = 'value', label = f'{region}')

#Plot for New Single-family Houses Sold (National)
house_sold = data[data['time_series_code'].str.contains("ASOLD_E_TOTAL_US_adj")]
pl_data(house_sold, 'Annual Rate for New Single-family Houses Sold', 'National Sales')

#convert date column to datetime
data['date'] = pd.to_datetime(data['date'])

#filter housing data to use
code = ['PERMITS_TOTAL_US','RATE_HOR_US', 'AUTHNOTSTD_TOTAL_US_adj','UNDERCONST_TOTAL_US']
filter = data[data['time_series_code'].isin(code)]

#create a pivot table
pivot = filter.pivot_table(index = 'date', columns = 'time_series_code', values = 'value', aggfunc = 'first')
pivot = pivot.apply(pd.to_numeric, errors = 'coerce').fillna(method = 'ffill')

corr = pivot.corr()

#create Correlation Matrix Visual
sns.heatmap(corr, annot = True, cmap = 'crest')
plt.title('Housing Correlation Matrix')


# creating dataframes for each one of variables that are wanted to look into


# Home ownership rate(%) from 2000 until 2017
x1 = data[data['time_series_code'].str.contains('RATE_HOR_US')]
x1 = x1[(x1.date > '2000-01-1')]

# Annual Rate for New Single-family Houses Sold(%) 2000 onwards
x2 = data[data['time_series_code'].str.contains('ASOLD_E_TOTAL_US')]

# Occupied Housing Units(in thousands) 2000 onwards
x3 = data[data['time_series_code'].str.contains('ESTIMATE_OCC_US')]

# Estimated Rented Houses(in thousands) 2000 onwards
x4 = data[data['time_series_code'].str.contains('ESTIMATE_RNTOCC_US')]

#Merging dataframes to get all values into one
combined_df = pd.merge(x1, x2, on='date', how='outer')
combined_df = pd.merge(combined_df, x3, on='date', how='outer')

# droping values that dont align with each other and adding revalent column names
combined_df = combined_df.dropna()
combined_df = combined_df.drop(columns=["time_series_code_x","time_series_code_y","time_series_code"], axis=1)
combined_df.rename(columns={"date":"date", "value_x": "Homeownership Rate", "value_y" : "Rate of Houses Sold", "value": "Estimated Occupaied Houses"}, inplace=True)

#merging, dropping, and renaming for additional column
combined_df = pd.merge(combined_df, x4, on='date', how='outer')
combined_df = combined_df.drop(columns=["time_series_code"], axis=1)
combined_df.rename(columns={"value": "Estimated Rented Houses"}, inplace=True)

#copying dataframe to make scatterplot
scatterdf = combined_df.copy()
scatterdf['date'] = pd.to_datetime(scatterdf['date'])

#changing date time into int of just the year
scatterdf['date'] = scatterdf['date'].dt.year

#subtracting a date so you can increment graph by just one year rather than datetime
start = 2000
scatterdf['date'] = scatterdf['date'] - start
print(scatterdf)
# print out 3d scatterplot showing rate of houses sold, homeownership rate,
# and the date together and relationship

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(scatterdf['Rate of Houses Sold'], scatterdf['date'], scatterdf['Homeownership Rate'], color = "green")

plt.show()

#Training data on only pre 2015 data to test against 2017 data for model accuracy
#9 values in test 58 in training
training = combined_df[combined_df['date'] < '2015-01-01']
test = combined_df[combined_df['date'] > '2015-01-01']

#spitting up the variables into predictors and predicted to see if there is correlation
training_predictors = training[["Homeownership Rate", "Rate of Houses Sold", "Estimated Rented Houses"]]
training_predicted = training["Estimated Occupaied Houses"]

test_predictors = test[["Homeownership Rate", "Rate of Houses Sold", "Estimated Rented Houses"]]
test_predicted = test["Estimated Occupaied Houses"]

#creating model and fitting model after we predict values based on test data
model = LinearRegression()
model.fit(training_predictors, training_predicted)

predicted_from_training_model = model.predict(test_predictors)
#shows mean squared error
mse = mean_squared_error(test_predicted,predicted_from_training_model)
print(mse)

#plot the data to make it easier to see
plt.scatter(test_predicted, predicted_from_training_model, color='green')
plt.plot(test_predicted, test_predicted, color='black')

plt.title('Actual vs. Predicted')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.show()

#show the rediuals and shows over time the model will degrade
residuals = test_predicted - predicted_from_training_model
print(residuals)

#getting r2 score
r2 = r2_score(test_predicted, predicted_from_training_model)
print(r2)

# A r2 score of .78 is regarded as strong linear fit for the model

