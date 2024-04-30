#reading in data and metadata
data = pd.read_csv("data.csv")
meta_data = pd.read_csv("metadata.csv")

#data quick stats
print(data.time_series_code.value_counts())
print(data.info())
print(data.describe())
print(data.shape)
print(data.size)
print(data.memory_usage)
