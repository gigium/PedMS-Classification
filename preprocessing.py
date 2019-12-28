import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# This function code the dataset classes in the 3 categories, 
# creating a list
def get_target_classes(col):
	y = []
	c = data.iloc[:, 0]

	for i in range(len(col)):
	    if "ADHD" in str(col.iloc[i]):
	        y.append(1)
	    elif "PMS" in str(col.iloc[i]):
	        y.append(2)
	    elif "HCPE" in str(col.iloc[i]):
	        y.append(0)
	return y


def read_dataT(file_name="longRNA_NGS.txt"):
	df = pd.read_csv(file_name, sep="\t", header=None)
	df = df.T
	return df


def standardize_dataset(df):
	r, df = df.iloc[0, :], df.iloc[1:,:]
	df.rename(columns=r, inplace=True)
	df = df.iloc[:, 1:]
	return df


def _PCA(df, targets, COMPONENTS=47):
	X = pd.DataFrame(StandardScaler().fit_transform(df))
	X.head()
	pca = PCA(n_components=COMPONENTS)
	principalComponents = pca.fit_transform(X)

	columns = []
	for i in range(COMPONENTS):
	    columns.append("PC"+str(i+1))

	principalDf = pd.DataFrame(data = principalComponents, columns=columns)
	targets = pd.DataFrame(targets)
	targets = targets.rename(columns={0:"targets"})
	finalDf = pd.concat([principalDf, targets], axis = 1)
	return finalDf




data = read_dataT()
y = get_target_classes(data)
data_S = standardize_dataset(data)

PCs = _PCA(data_S, y)
print(PCs.head())