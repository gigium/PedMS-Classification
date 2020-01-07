from oversampling import randomOverSampling, SMOTEOverSampling
from feature_selection import (recursiveFElimination, lassoFSelect,
								 lowMeanElimination, lowVarianceElimination, read_data)
from classification import svm, decision_tree
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




def runExperiment(DIR, _exp, run_arg):
	mlflow.set_experiment(DIR+"_nome esperimento 2_")
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
		



def exp_example(train, test, kernel):
	# pre_processing
	over_sampled_train = randomOverSampling(train)
	keep = lowVarianceElimination(over_sampled_train, 0.8)
	train = Standardization(over_sampled_train[keep])
	test = Standardization(test[keep])
	return svm(train,test,kernel)



