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
