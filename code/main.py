import os, sys
from feature_selection import read_data
from experiments import runExperiment, experiment3_2
import mlflow


def main():
	DIR = sys.argv[1]
	runExperiment(DIR, experiment3_2, [0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99])


# python main.py .\kFold
if __name__== "__main__":
	main()
