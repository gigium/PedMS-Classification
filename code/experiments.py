from oversampling import randomOverSampling, SMOTEOverSampling
from feature_selection import (univariateFSelect,decisionTreeFSelect,recursiveFElimination, 
								lassoFSelect, univariateFSelect,
								lowMeanElimination, lowVarianceElimination, read_data)
from classification import svm, decision_tree, randomForest, feedForwardNN
from standardization import Standardization, MinMaxScaler

import mlflow
import numpy as np
import os


def getFold(DIR):
	n_files = int(len(os.listdir(DIR))/2) 

	fold = {"train":[],"test":[]}

	for i in range(n_files):
		# print("\n______________________________fold %s_______________________________\n" %str(i))

		fold["train"].append(read_data(DIR+"/train_"+str(i)+".txt"))
		fold["test"].append(read_data(DIR+"/test_"+str(i)+".txt"))

	return fold




def runExperiment(DIR, _exp, run_arg, experiment_name=""):
	mlflow.set_experiment(DIR+" "+_exp.__name__+experiment_name)
	fold = getFold(DIR)

	for arg in run_arg:
		with mlflow.start_run(run_name="argument " + str(arg), nested=True):

			results = {"accuracy":[],"f1":[],"precision":[],"recall":[],}

			for i in range(len(fold["train"])):
				print("\n______________________________fold %s_______________________________\n" %str(i))
				a,f,p,r = _exp(fold["train"][i], fold["test"][i], arg)
				results["accuracy"].append(a)
				results["f1"].append(f)
				results["precision"].append(p)
				results["recall"].append(r)

			mlflow.log_metric("accuracy", np.mean(results["accuracy"]))
			mlflow.log_metric("f1",  np.mean(results["f1"]))
			mlflow.log_metric("precision", np.mean(results["precision"]))
			mlflow.log_metric("recall", np.mean(results["recall"]))
		



def random_forest_depth_exp_SM_LV_ST_RF(train, test, max_depth):
	over_sampled_train = SMOTEOverSampling(train)
	keep = lowVarianceElimination(over_sampled_train, 0.8)
	train = Standardization(over_sampled_train[keep])
	test = Standardization(test[keep])
	return randomForest(train, test, max_depth=max_depth)


#best k=1000
def experiment3_0_1(train, test, k):
	over_sampled_train = SMOTEOverSampling(train)

	keep = lowVarianceElimination(over_sampled_train,0.8)
	keep = univariateFSelect(over_sampled_train[keep],k)	

	train = Standardization(over_sampled_train[keep])
	test = Standardization(test[keep])

	return svm(train,test)




#best k=1000
def experiment3_0_2(train, test, k):
	over_sampled_train = SMOTEOverSampling(train)

	keep = lowVarianceElimination(over_sampled_train, 0.8)
	keep = decisionTreeFSelect(over_sampled_train[keep], k)	

	train = Standardization(over_sampled_train[keep])
	test = Standardization(test[keep])

	return svm(train,test)



# variance [0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99] no difference
def experiment3_1(train, test, variance):
	over_sampled_train = SMOTEOverSampling(train)

	keep = lowVarianceElimination(over_sampled_train, variance)
	keep = univariateFSelect(over_sampled_train[keep],1000)

	train = Standardization(over_sampled_train[keep])
	test = Standardization(test[keep])

	return svm(train,test)



# variance [0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99] best 0.8 with accurcy 0.7
def experiment3_2(train, test, variance):
	over_sampled_train = SMOTEOverSampling(train)

	keep = lowVarianceElimination(over_sampled_train, variance)
	keep = decisionTreeFSelect(over_sampled_train[keep],1000)

	train = Standardization(over_sampled_train[keep])
	test = Standardization(test[keep])

	return svm(train,test)



def experiment4(train, test, variance):
	over_sampled_train = SMOTEOverSampling(train)
	keep = lowVarianceElimination(over_sampled_train, variance)
	keep = lassoFSelect(over_sampled_train[keep])
	train = Standardization(over_sampled_train[keep])
	test = Standardization(test[keep])
	return svm(train, test)



def experiment2_1(train, test, f):
	over_sampled_train = SMOTEOverSampling(train)

	keep = univariateFSelect(over_sampled_train, 1000)
	keep = f(over_sampled_train[keep])

	train = Standardization(over_sampled_train[keep])
	test = Standardization(test[keep])

	return svm(train, test)



def experiment2_2(train, test, f):
	over_sampled_train = SMOTEOverSampling(train)

	keep = decisionTreeFSelect(over_sampled_train, 1000)
	keep = f(over_sampled_train[keep])

	train = Standardization(over_sampled_train[keep])
	test = Standardization(test[keep])

	return svm(train, test)



def experiment1(train, test, f):
	over_sampled_train = SMOTEOverSampling(train)

	keep = f(over_sampled_train)

	train = Standardization(over_sampled_train[keep])
	test = Standardization(test[keep])

	return svm(train, test)



def experiment7(train, test, f):
	over_sampled_train = SMOTEOverSampling(train)

	keep = f(over_sampled_train)

	train = Standardization(over_sampled_train[keep])
	test = Standardization(test[keep])

	return feedForwardNN(train, test)
