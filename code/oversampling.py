from imblearn.over_sampling import RandomOverSampler,SMOTE
import pandas as pd
from collections import Counter
import mlflow


def randomOverSampling(df):
	X = df.iloc[:,:-1]  
	y = df.iloc[:,-1]  
	ros = RandomOverSampler(random_state=0)
	X_resampled, y_resampled = ros.fit_resample(X, y)
	X_resampled['target'] = y_resampled
	print("\n")
	print("random oversampling ... ")
	print("from ... ", sorted(Counter(df["target"]).items()))
	print("to ... ", sorted(Counter(y_resampled).items()))

	mlflow.log_param("OVERSAMPLING-RANDOM classes resampled", sorted(Counter(y_resampled).items()))

	return X_resampled 



def SMOTEOverSampling(df, neighbors=2):
	X = df.iloc[:,:-1]  
	y = df.iloc[:,-1]  
	ros = SMOTE(random_state=0, k_neighbors=neighbors)
	X_resampled, y_resampled = ros.fit_resample(X, y)
	X_resampled['target'] = y_resampled
	print("\n")
	print("SMOTE oversampling ... ")
	print("from ... ", sorted(Counter(df["target"]).items()))
	print("to ... ", sorted(Counter(y_resampled).items()))

	mlflow.log_param("OVERSAMPLING-SMOTE classes resampled", sorted(Counter(y_resampled).items()))

	return X_resampled 