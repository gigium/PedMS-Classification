from oversampling import randomOverSampling, SMOTEOverSampling
from feature_selection import (recursiveFElimination, lassoFSelect,
								 lowMeanElimination, lowVarianceElimination)
from classification import svm, decision_tree, feedForwardNN

import mlflow





def choose_variance_treshold(train_, test_, fold):
	treshold = 0

	while treshold < 1:
		# mlflow.set_tag("variance treshold", treshold)

		train, test = train_, test_
		with mlflow.start_run(run_name="variance treshold " + str(treshold), nested=True):
			mlflow.log_param("fold", fold)

			over_sampled_train = randomOverSampling(train)
			
			keep = lowVarianceElimination(over_sampled_train, treshold)

			# keep = lassoFSelect(over_sampled_train[keep])
			
			train = over_sampled_train[keep]
			test = test[keep]
			
			svm(train,test)
			treshold+=.1