import numpy as np
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import os
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



def  createFolder(df,split_dict,k,path="./kFold"):
	access_rights = 0o777

	try:
	    os.mkdir(path, access_rights)
	except OSError:
	    print ("Creation of the directory %s failed" % path)
	else:
	    print ("Successfully created the directory %s" % path)

	train = split_dict["train"]
	test = split_dict["test"]

	for i in range(len(train)):
		df.iloc[train[i]].to_csv(path+"/train_"+str(i)+".txt",sep="\t")
		df.iloc[test[i]].to_csv(path+"/test_"+str(i)+".txt",sep="\t")



def main():
	data = read_data(sys.argv[1])
	k=int(sys.argv[2])
	split=stratifiedKfold(data,k)
	print(split)
	createFolder(data,split,k)



# python stratified_k_fold.py .\data\new_data.txt 3
if __name__== "__main__":
  main()
