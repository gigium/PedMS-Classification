import sys

import mlflow
import mlflow.sklearn

from feature_selection import lowVarianceElimination, correlationFElimination, read_data


def main():
	data = read_data(sys.argv[1])
	d = lowVarianceElimination(data, 0.95)
	d1 = correlationFElimination(d, 0.8)
	print(d1.head())
	# mlflow.set_experiment("lowVarianceElimination")
	# k = 0
	# while k < 1:
	# 	mlflow.log_param("treshold", k)
	# 	d = lowVarianceElimination(data, k)
	# 	var = np.mean(d.var(axis=1))

	# 	mlflow.log_metric("treshold", k)
	# 	data_amnt = len(d.columns)*len(d.index)
	# 	mlflow.log_metric("variance", var)
	# 	mlflow.log_metric("cols", len(d.columns))
	# 	k+= 0.05


# after the run of the script, lunch 'mlflow ui' command 
# and go 'to http://localhost:5000' to see the ui
# python find_best_features.py .\data\new_data.txt
if __name__== "__main__":
	main()
