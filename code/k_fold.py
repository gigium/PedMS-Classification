import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import LeaveOneOut
import pandas as pd
import os
import sys
from feature_selection import read_data



def leaveOneOut(df):
	# target and data selection
	y=df.iloc[:,-1]
	X=df.iloc[:,:-1]
	y=y.to_numpy()

	loo = LeaveOneOut()
	loo.get_n_splits(X,y)

	
	split={
	"train":[],
	"test":[]
	}
	
	for train_index, test_index in loo.split(X, y):
		print("loo sizes", len(train_index), len(test_index))
		split["train"].append(train_index)
		split["test"].append(test_index)
	
	print("leave one out ... created ", len(split["train"]))

	return split



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
	
	print("stratified k fold ... k = ", k)

	return split



def  createFolder(df, split_dict, path="./kFold"):
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
	if sys.argv[2] == "stratified":
		k=int(sys.argv[3])
		split=stratifiedKfold(data,k)
		if sys.argv[4]:
			createFolder(data, split, path=sys.argv[4])
		else:
			createFolder(data, split)

	elif sys.argv[2] == "loo":
		split=leaveOneOut(data)
		createFolder(data, split, path="./loo")



# python k_fold.py .\data\new_data.txt [stratified | loo] {k}
if __name__== "__main__":
  main()
