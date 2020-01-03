from imblearn.over_sampling import RandomOverSampler,SMOTE
import pandas as pd


def randomOverSampling(df):
	X = df.iloc[:,:-1]  
	y = df.iloc[:,-1]  
	ros = RandomOverSampler(random_state=0)
	X_resampled, y_resampled = ros.fit_resample(X, y)
	X_resampled['target'] = y_resampled
	return X_resampled 



def SMOTEOverSampling(df):
	X = df.iloc[:,:-1]  
	y = df.iloc[:,-1]  
	ros = SMOTE(random_state=0)
	X_resampled, y_resampled = ros.fit_resample(X, y)
	X_resampled['target'] = y_resampled
	return X_resampled 