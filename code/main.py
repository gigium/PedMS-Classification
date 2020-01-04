import sys, os
import numpy as np
import mlflow
import mlflow.sklearn

from feature_selection import (recursiveFElimination, lassoFSelect,
								 read_data, 
								 lowMeanElimination, lowVarianceElimination)
from oversampling import randomOverSampling, SMOTEOverSampling
from classification import svm


from sklearn import  linear_model
from sklearn.metrics import mean_squared_error, r2_score

from collections import Counter


def main():
	# path joining version for other paths
	DIR = './kFold'
	n_files = int(len(os.listdir(DIR))/2) 
	
	for i in range(n_files):
		train = read_data("./kFold/train_"+str(i)+".txt")
		test = read_data("./kFold/test_"+str(i)+".txt")

		over_sampled_train = SMOTEOverSampling(train)

		keep = lowVarianceElimination(over_sampled_train, .99)
		keep = lassoFSelect(over_sampled_train[keep])
		
		train = over_sampled_train[keep]
		test = test[keep]
		print("test classes ", sorted(Counter(test["target"]).items()))
		
		report=svm(train,test)
		print(report)


# after the run of the script, lunch 'mlflow ui' command 
# and go 'to http://localhost:5000' to see the ui
# python main.py 
if __name__== "__main__":
	main()
