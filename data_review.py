#reading in data and metadata
data = pd.read_csv("data.csv")
meta_data = pd.read_csv("metadata.csv")

#printing information about data and meta data
print(data.time_series_code.value_counts())
