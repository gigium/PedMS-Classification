import sys
import numpy as np
import mlflow
import mlflow.sklearn

from feature_selection import recursiveFElimination, lassoFSelect, read_data
from oversampling import randomOverSampling, SMOTEOverSampling
from classification import svm


from sklearn import  linear_model
from sklearn.metrics import mean_squared_error, r2_score


def main():
	
	train = read_data("./kFold/train_"+str(1)+".txt")
	test = read_data("./kFold/test_"+str(1)+".txt")
	report=svm(train,test)
	print(report)


# after the run of the script, lunch 'mlflow ui' command 
# and go 'to http://localhost:5000' to see the ui
# python main.py 
if __name__== "__main__":
	main()
