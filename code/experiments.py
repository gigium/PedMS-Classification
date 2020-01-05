from oversampling import randomOverSampling, SMOTEOverSampling
from feature_selection import (recursiveFElimination, lassoFSelect,
								 lowMeanElimination, lowVarianceElimination)
from classification import svm, decision_tree, feedForwardNN



# Set up various experiments with the same methods, but changing parameters e.g.variance treshold ...
def exp1(train, test):
	
	over_sampled_train = randomOverSampling(train)
	
	keep = lowVarianceElimination(over_sampled_train, .99)
	keep = lassoFSelect(over_sampled_train[keep])
	
	train = over_sampled_train[keep]
	test = test[keep]
	
	feedForwardNN(train,test, lossF='cosine')