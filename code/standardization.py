import pandas as pd
import numpy as np

from sklearn import preprocessing

# standardizing data the mean will be zero and the standard deviation one.
def Standardization(data):

	target=data.iloc[:,-1]
	X=data.iloc[:,:-1]

	indexes = X.index
	columns = X.columns

	std_scale = preprocessing.StandardScaler().fit(X)
	std_X = std_scale.transform(X)

	final_data = pd.DataFrame(data=std_X)
	final_data.columns = columns
	final_data.index = indexes
	final_data['target'] = target

	return final_data

#  the data is scaled to a fixed range - usually 0 to 1
def MinMaxScaler(data, min_value=0, max_value=1):

	target=data.iloc[:,-1]
	X=data.iloc[:,:-1]

	indexes = X.index
	columns = X.columns

	minmax_scale = preprocessing.MinMaxScaler(feature_range=(0,1)).fit(X)
	minmax_X = minmax_scale.transform(X)

	final_data = pd.DataFrame(data=minmax_X)
	final_data.columns = columns
	final_data.index = indexes
	final_data['target'] = target

	
	return final_data




