import os, sys
from feature_selection import read_data
from experiments import neuralNet_epoch_exp_SM_LV_ST_NN, random_forest_depth_exp_SM_LV_ST_RF, pca_n_components_exp

from experiments import (runExperiment, experiment1, experiment2, experiment2_1,
						 	experiment3_0_1, experiment3_0_2, experiment3_1, experiment3_2, experiment4,
							experiment5, experiment6, experiment6_1, 
							experiment7, experiment8, experiment8_1,
							experiment9, experiment10, experiment10_1,
							experiment11, experiment12, experiment12_1,
							experiment13, experiment14, experiment14_1,
							majority_vote_exp, majority_vote_exp_1, majority_vote_exp_2)

from feature_selection import (correlationFElimination, lassoFSelect, recursiveFElimination, 
								lowMeanElimination, lowVarianceElimination, univariateFSelect, 
								decisionTreeFSelect)

from sklearn.feature_selection import chi2 , f_classif,  mutual_info_classif

import mlflow


def main():
	DIR = sys.argv[1]
	
	# runExperiment(DIR, experiment1, [lowMeanElimination, lowVarianceElimination, univariateFSelect, decisionTreeFSelect, lassoFSelect])

	# runExperiment(DIR, experiment2, [correlationFElimination, lassoFSelect, recursiveFElimination])
	# runExperiment(DIR, experiment2_1, [correlationFElimination, lassoFSelect, recursiveFElimination])
	# runExperiment(DIR, experiment2_1, [recursiveFElimination]) k 500
	# runExperiment(DIR, experiment2_2, [recursiveFElimination])

	# runExperiment(DIR, experiment3_0_1, [10, 100, 1000, 10000, 20000])
	# runExperiment(DIR, experiment3_0_1, [100, 200, 300, 400, 500, 600, 700, 800])
	# runExperiment(DIR, experiment3_0_2, [10, 100, 1000, 10000, 20000])	
	# runExperiment(DIR, experiment3_0_1, [100, 200, 300, 400, 500, 600, 700, 800])

	# runExperiment(DIR, experiment3_1, [.1, .3, .5, .7, .8, .9, .95, .99])
	# runExperiment(DIR, experiment3_2, [.1, .3, .5, .7, .8, .9, .95, .99])

	# runExperiment(DIR, experiment4, [.1, .3, .5, .7, .8, .9, .95, .99])

	# runExperiment(DIR, experiment5, [lowMeanElimination, lowVarianceElimination, univariateFSelect, decisionTreeFSelect, lassoFSelect])
	# runExperiment(DIR, experiment6, [lassoFSelect, correlationFElimination, recursiveFElimination])
	# runExperiment(DIR, experiment6_1, [lassoFSelect, correlationFElimination, recursiveFElimination])

	# runExperiment(DIR, experiment7, [lassoFSelect, univariateFSelect, lowVarianceElimination, lowMeanElimination, decisionTreeFSelect])
	# runExperiment(DIR, experiment8, [lassoFSelect, correlationFElimination, recursiveFElimination])
	# runExperiment(DIR, experiment8_1, [lassoFSelect, correlationFElimination, recursiveFElimination])

	# runExperiment(DIR, experiment9, [lassoFSelect, univariateFSelect, lowVarianceElimination, lowMeanElimination, decisionTreeFSelect])
	# runExperiment(DIR, experiment10, [lassoFSelect, correlationFElimination, recursiveFElimination])
	# runExperiment(DIR, experiment10_1, [lassoFSelect, correlationFElimination, recursiveFElimination])
	
	# runExperiment(DIR, experiment11, [lassoFSelect, univariateFSelect, lowVarianceElimination, lowMeanElimination, decisionTreeFSelect])
	# runExperiment(DIR, experiment12, [lassoFSelect, correlationFElimination, recursiveFElimination])
	# runExperiment(DIR, experiment12_1, [lassoFSelect, correlationFElimination, recursiveFElimination])

	# runExperiment(DIR, experiment13, [lassoFSelect, univariateFSelect, lowVarianceElimination, lowMeanElimination, decisionTreeFSelect])
	# runExperiment(DIR, experiment14, [lassoFSelect, correlationFElimination, recursiveFElimination])
	# runExperiment(DIR, experiment14_1, [lassoFSelect, correlationFElimination, recursiveFElimination])



	runExperiment(DIR, majority_vote_exp, [lassoFSelect, univariateFSelect, lowVarianceElimination, lowMeanElimination, decisionTreeFSelect])
	runExperiment(DIR, majority_vote_exp_1, [lassoFSelect, correlationFElimination, recursiveFElimination])
	runExperiment(DIR, majority_vote_exp_2, [lassoFSelect, correlationFElimination, recursiveFElimination])


	# runExperiment(DIR, pca_n_components_exp, [2, 4, 8])

	# runExperiment(DIR, neuralNet_epoch_exp_SM_LV_ST_NN, [5, 10, 20, 30, 40, 50, 60, 70, 80, 90], "_1")
	# runExperiment(DIR, random_forest_depth_exp_SM_LV_ST_RF, [1, 3, 5, 7, 10, 20, 30])
	#runExperiment(DIR, univariate_function_exp_SM_UFS_ST_SVM, [f_classif, chi2,  mutual_info_classif],"univariate_score_func" )




# python main.py .\kFold
if __name__== "__main__":
	main()
