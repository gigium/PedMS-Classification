import numpy as np
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import sys



def read_data(file_name):
	df = pd.read_csv(file_name, sep="\t", index_col=0)
	return df



def stratifiedKfold(df, k=5):
	# target and data selection
	y=df.iloc[:,-1]
	X=df.iloc[:,:-1]
	y=y.to_numpy()

	skf =StratifiedKFold(k)
	skf.get_n_splits(X,y)
	
	split={
    "train":[],
    "test":[]
	}
	
	for train_index, test_index in skf.split(X, y):
	    split["train"].append(train_index)
	    split["test"].append(test_index)
	
	return split



def main():
	data = read_data(sys.argv[1])
	split=stratifiedKfold(data)
	print(split)


# python stratified_k_fold.py .\data\new_data.txt
if __name__== "__main__":
  main()
