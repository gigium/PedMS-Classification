from oversampling import randomOverSampling, SMOTEOverSampling
from feature_selection import (recursiveFElimination, lassoFSelect,
								 lowMeanElimination, lowVarianceElimination)
from classification import svm, decision_tree
from standardization import Standardization, MinMaxScaler

import mlflow


def svmKernel_RO_LVE_S_SVM(train_,test_):
	kernels= ["linear", "poly", "rbf", "sigmoid", "precomputed"]
	# pre_processing
	over_sampled_train = randomOverSampling(train_)
	keep = lowVarianceElimination(over_sampled_train, 0.8)
	train = Standardization(over_sampled_train[keep])
	test = Standardization(test_[keep])


	for kernel in kernels:

		svm(train,test,kernel)






def choose_variance_treshold(train_, test_):
	treshold = 0

	while treshold < 1:
		# mlflow.set_tag("variance treshold", treshold)

		train, test = train_, test_
		# with mlflow.start_run(run_name="variance treshold " + str(treshold), nested=True):

		over_sampled_train = randomOverSampling(train)
		
		keep = lowVarianceElimination(over_sampled_train, treshold)

		# keep = lassoFSelect(over_sampled_train[keep])
		
		train = over_sampled_train[keep]
		test = test[keep]
		
		svm(train,test)
		treshold+=.1