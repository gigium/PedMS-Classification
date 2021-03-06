import pandas as pd
import sys


def get_target_classes(data):
	y = []
	col = data.iloc[:, 0]

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


def main():
	data = read_dataT(sys.argv[1])
	y = get_target_classes(data)
	data_S = standardize_dataset(data)
	data_S["target"] = y

	data_S.to_csv(sys.argv[2], sep="\t")

 
# python preprocessing.py .\data\longRNA_NGS.txt .\data\new_data.txt
if __name__== "__main__":
  main()



