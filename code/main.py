import os
from feature_selection import read_data
from experiments import choose_variance_treshold
import mlflow


def executeExpinFold(DIR ,experiment_name, experiment_f):
	mlflow.set_experiment(DIR+", "+experiment_name)
	n_files = int(len(os.listdir(DIR))/2) 

	for i in range(n_files):
		with mlflow.start_run(run_name=experiment_name + " fold " + str(i)):
			print("\n______________________________fold %s_______________________________\n" %str(i))

			train = read_data("./kFold/train_"+str(i)+".txt")
			test = read_data("./kFold/test_"+str(i)+".txt")

			experiment_f(train, test, i)



def main():
	executeExpinFold("./kFold" ,"randomOverSampling, lowVarianceElimination, SVM", choose_variance_treshold)



# after the run of the script, lunch 'mlflow ui' command 
# and go 'to http://localhost:5000' to see the ui


# python main.py
if __name__== "__main__":
	main()
