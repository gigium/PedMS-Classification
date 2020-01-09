import os, sys
from feature_selection import read_data
from experiments import neuralNet_epoch_exp_SM_LV_ST_NN, random_forest_depth_exp_SM_LV_ST_RF
from experiments import (runExperiment, experiment3_0_1, experiment3_0_2, experiment2_1, experiment2_2, experiment1, experiment3_1,
						experiment5, experiment6, experiment6_1, experiment3_2, experiment4, experiment7, experiment8, experiment8_1)

from feature_selection import (correlationFElimination, lassoFSelect, recursiveFElimination, 
								lowMeanElimination, lowVarianceElimination, univariateFSelect, 
								decisionTreeFSelect)
import mlflow


def main():
	DIR = sys.argv[1]
	

	# runExperiment(DIR, experiment1, [lowMeanElimination, lowVarianceElimination, univariateFSelect, decisionTreeFSelect, lassoFSelect])

	# runExperiment(DIR, experiment2_1, [correlationFElimination, lassoFSelect])
	# runExperiment(DIR, experiment2_2, [correlationFElimination, lassoFSelect])
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
	runExperiment(DIR, experiment6, [lassoFSelect, correlationFElimination, recursiveFElimination])
	runExperiment(DIR, experiment6_1, [lassoFSelect, correlationFElimination, recursiveFElimination])

	# runExperiment(DIR, experiment7, [lassoFSelect, univariateFSelect, lowVarianceElimination, lowMeanElimination, decisionTreeFSelect])
	# runExperiment(DIR, experiment8, [lassoFSelect, correlationFElimination, recursiveFElimination])
	# runExperiment(DIR, experiment8_1, [lassoFSelect, correlationFElimination, recursiveFElimination])

	# runExperiment(DIR, neuralNet_epoch_exp_SM_LV_ST_NN, [5, 10, 20, 30, 40, 50, 60, 70, 80, 90], "_1")
	# runExperiment(DIR, random_forest_depth_exp_SM_LV_ST_RF, [1, 3, 5, 7, 10, 20, 30])





# python main.py .\kFold
if __name__== "__main__":
	main()
