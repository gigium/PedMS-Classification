import sys
import numpy as np
import mlflow
import mlflow.sklearn

from feature_selection import lowVarianceElimination, correlationFElimination, read_data
from oversampling import randomOverSampling, SMOTEOverSampling

def main():
	data = read_data(sys.argv[1])

# after the run of the script, lunch 'mlflow ui' command 
# and go 'to http://localhost:5000' to see the ui
# python find_best_features.py .\data\new_data.txt
if __name__== "__main__":
	main()
