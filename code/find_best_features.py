import sys

import mlflow
import mlflow.sklearn

from feature_selection import lowVarianceElimination, correlationFElimination, read_data


def main():
	data = read_data(sys.argv[1])
	mlflow.set_experiment("variance_decision")
	k=0
	while(k<1):
		print(k) 
		d = lowVarianceElimination(data,k)
		row, columns=d.shape
		mlflow.log_metric("k",k)
		mlflow.log_metric("columns",columns)
		k+=0.05

	



# after the run of the script, lunch 'mlflow ui' command 
# and go 'to http://localhost:5000' to see the ui
# python find_best_features.py .\data\new_data.txt
if __name__== "__main__":
	main()
