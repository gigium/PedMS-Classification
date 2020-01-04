import numpy as np
import pandas as pd

from feature_selection import read_data


from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from sklearn.metrics import classification_report, confusion_matrix

import warnings



def svm(train,test,kernel='rbf'):
	print("svm ... ")
	print("kernel function : ", kernel)

	y_train=train.iloc[:,-1]
	X_train=train.iloc[:,:-1]

	y_test=test.iloc[:,-1]
	X_test=test.iloc[:,:-1]



	svclassifier = SVC(kernel='rbf')
	svclassifier.fit(X_train, y_train)

	y_pred = svclassifier.predict(X_test)

	with warnings.catch_warnings():
		# ignore all caught warnings
		warnings.filterwarnings("ignore")
		report = classification_report(y_test,y_pred)
	
	return report




# python classification.py




