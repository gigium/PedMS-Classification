import sys
import numpy as np
import mlflow
import mlflow.sklearn

from feature_selection import recursiveFElimination, lassoFSelect, read_data
from oversampling import randomOverSampling, SMOTEOverSampling
from classification import svm, decision_tree


from sklearn import  linear_model
from sklearn.metrics import mean_squared_error, r2_score


def main():
	
	train = read_data("./kFold/train_"+str(1)+".txt")
	test = read_data("./kFold/test_"+str(1)+".txt")
	report_svm=svm(train,test)
	report_decision_tree=decision_tree(train,test)
	print("report_svm ....\n", report_svm )
	print("report_decision_tree ....\n", report_decision_tree )




# after the run of the script, lunch 'mlflow ui' command 
# and go 'to http://localhost:5000' to see the ui
# python main.py 
if __name__== "__main__":
	main()
