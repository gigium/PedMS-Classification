import os, sys
from feature_selection import read_data
from experiments import runExperiment, experiment3_0_1
import mlflow


def main():
	DIR = sys.argv[1]
	runExperiment(DIR, experiment3_0_1, [10,100,1000,10000,20000])


# python main.py .\kFold
if __name__== "__main__":
	main()
