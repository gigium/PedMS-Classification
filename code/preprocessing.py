import pandas as pd
import sys


def get_target_classes(data):
	y = []
	col = data.iloc[:, 0]

	for i in range(len(col)):
	    if "ADHD" in str(col.iloc[i]):
	        y.append(1)
	    elif "PMS" in str(col.iloc[i]):
	        y.append(2)
	    elif "HCPE" in str(col.iloc[i]):
	        y.append(0)
	return y


def read_dataT(file_name="longRNA_NGS.txt"):
	df = pd.read_csv(file_name, sep="\t", header=None)
	df = df.T
	return df


def standardize_dataset(df):
	r, df = df.iloc[0, :], df.iloc[1:,:]
	df.rename(columns=r, inplace=True)
	df = df.iloc[:, 1:]
	return df


def main():
	data = read_dataT(sys.argv[1])
	y = get_target_classes(data)
	data_S = standardize_dataset(data)
	targets = pd.DataFrame(y).rename(columns={0:"targets"}, inplace=True)
	final_data = pd.concat([data_S, targets], axis = 1)
	print(final_data.head())
	final_data.to_csv(sys.argv[2], sep="\t")

 
# python preprocessing.py .\data\longRNA_NGS.txt .\data\new_data.txt
if __name__== "__main__":
  main()


#mlflow.log_metric("variance_pc1",variance[1])# log of metric
#mlflow.log_metric("variance_pc2",variance[2])# log of metric
#mlflow.log_metric("variance_pc2",variance[3])# log of metric



# after the run of the script, lunch 'mlflow ui' command 
# and go 'to http://localhost:5000' to see the ui
