import numpy as np
import pandas as pd

from feature_selection import read_data


from sklearn import svm
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


from sklearn.metrics import (accuracy_score, f1_score, precision_score, 
								recall_score, classification_report, 
								confusion_matrix)

import mlflow

import warnings


def split(train,test):

	y_train=train.iloc[:,-1]
	X_train=train.iloc[:,:-1]

	y_test=test.iloc[:,-1]
	X_test=test.iloc[:,:-1]

	return X_train, y_train, X_test, y_test


def reporter(y_test,y_pred):
	with warnings.catch_warnings():
		# ignore all caught warnings
		warnings.filterwarnings("ignore")
		report = classification_report(y_test,y_pred)	
		mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
		mlflow.log_metric("f1", f1_score(y_test, y_pred, average="macro"))
		mlflow.log_metric("precision", precision_score(y_test, y_pred, average="macro"))
		mlflow.log_metric("recall", recall_score(y_test, y_pred, average="macro"))
	return report
	

def svm(train,test,kernel='rbf'):
	print("svm ... ")
	print("kernel function : ", kernel)

	X_train, y_train, X_test, y_test = split(train,test)

	svclassifier = SVC(kernel='rbf')
	svclassifier.fit(X_train, y_train)

	y_pred = svclassifier.predict(X_test)
	
	report = reporter(y_test,y_pred)

	print(report)


def decision_tree(train,test):
	print("decision_tree ... ")

	X_train, y_train, X_test, y_test = split(train,test)

	classifier = DecisionTreeClassifier()
	classifier.fit(X_train, y_train)	

	y_pred = classifier.predict(X_test)

	report = reporter(y_test,y_pred)
	print(report)



