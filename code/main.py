import sys, os
import numpy as np

from feature_selection import (recursiveFElimination, lassoFSelect,
								 read_data, 
								 lowMeanElimination, lowVarianceElimination)
from oversampling import randomOverSampling, SMOTEOverSampling

from classification import svm, decision_tree, feedForwardNN

from standardization import Standardization, MinMaxScaler

import mlflow


def main():

	DIR = './kFold'

	n_files = int(len(os.listdir(DIR))/2) 

	mlflow.set_experiment(DIR+", SMOTE, lowVarianceElimination + lassoFSelect, NN")

	for i in range(n_files):
		with mlflow.start_run():
			
			mlflow.log_param("fold", i)

			print("\n______________________________fold %s_______________________________\n" %str(i))

			train = read_data("./kFold/train_"+str(i)+".txt")
			test = read_data("./kFold/test_"+str(i)+".txt")

			standard_train = Standardization(train)
			minmax_train = MinMaxScaler(train)


			over_sampled_train = SMOTEOverSampling(train)

			keep = lowVarianceElimination(over_sampled_train, .99)
			keep = lassoFSelect(over_sampled_train[keep])
			
			train = over_sampled_train[keep]
			test = test[keep]
			
			feedForwardNN(train,test, lossF='cosine')


# after the run of the script, lunch 'mlflow ui' command 
# and go 'to http://localhost:5000' to see the ui


# python main.py
if __name__== "__main__":
	main()
