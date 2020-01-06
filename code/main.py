import os, sys
from feature_selection import read_data
from experiments import choose_variance_treshold, svmKernel_RO_LVE_S_SVM
import mlflow


def executeExpinFold(DIR ,experiment_name, experiment_f):
	mlflow.set_experiment(DIR+", "+experiment_name)
	n_files = int(len(os.listdir(DIR))/2) 

	for i in range(n_files):
		with mlflow.start_run(run_name=experiment_name + " fold " + str(i)):
			print("\n______________________________fold %s_______________________________\n" %str(i))

			train = read_data(DIR+"/train_"+str(i)+".txt")
			test = read_data(DIR+"/test_"+str(i)+".txt")
			mlflow.log_param("fold", i)

			experiment_f(train, test)



def main():
	DIR = sys.argv[1]
	executeExpinFold(DIR ,"svmKernel_RO_LVE_S_SVM", svmKernel_RO_LVE_S_SVM)



# after the run of the script, lunch 'mlflow ui' command 
# and go 'to http://localhost:5000' to see the ui


# python main.py .\kFold
if __name__== "__main__":
	main()
