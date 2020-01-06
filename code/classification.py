import numpy as np
import pandas as pd

from feature_selection import read_data

#from keras.models import Sequential
#from keras.layers import Dense

from sklearn import svm
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (accuracy_score, f1_score, precision_score, 
								recall_score, classification_report, 
								confusion_matrix)

import mlflow

import warnings

from collections import Counter



def split(train,test):
	y_train=train.iloc[:,-1]
	X_train=train.iloc[:,:-1]
	y_test=test.iloc[:,-1]
	X_test=test.iloc[:,:-1]
	print("\n")
	print("train classes ", sorted(Counter(y_train).items()))
	print("test classes ", sorted(Counter(y_test).items()))

	mlflow.log_param("test classes", sorted(Counter(y_test).items()))

	return X_train, y_train, X_test, y_test



def reporter(y_test,y_pred):
	print("_____________________________________________________________________\n")	
	print("test: ", y_test, "\npred: ", y_pred)
	print("_____________________________________________________________________\n")
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
	print("\n")
	print("svm ... ")
	print("kernel function : ", kernel)

	mlflow.log_param("CLASSIFICATION-SVM kernel function", kernel)
	
	X_train, y_train, X_test, y_test = split(train,test)

	svclassifier = SVC(kernel='rbf')
	svclassifier.fit(X_train, y_train)

	y_pred = svclassifier.predict(X_test)
	report = reporter(y_test.to_numpy(),y_pred)

	print(report)



def decision_tree(train,test):
	print("\n")
	print("decision_tree ... ")

	X_train, y_train, X_test, y_test = split(train,test)

	classifier = DecisionTreeClassifier()
	classifier.fit(X_train, y_train)	

	y_pred = classifier.predict(X_test)

	report = reporter(y_test,y_pred)
	print(report)



def feedForwardNN(train, test,
					 layer1=64, layer2=1, 
					 activation1='relu', 
					 activation2='sigmoid',
					 lossF='mean_squared_error',
					 optimizerF='rmsprop',
					 metrics=['accuracy'],
					 epochs=20):

	print("\n")
	print("feedForwardNN ... ")

	print("loss function : ", lossF)
	print("optimizer function : ", optimizerF)

	mlflow.log_param("CLASSIFICATION-NN loss function", lossF)
	mlflow.log_param("CLASSIFICATION-NN optimizer function", optimizerF)

	X_train, y_train, X_test, y_test = split(train,test)

	model = Sequential() 
	model.add(Dense(layer1, input_dim=len(X_train.columns), activation=activation1)) 
	model.add(Dense(layer2, activation=activation2)) 

	model.compile(loss=lossF,
	              optimizer=optimizerF,
	              metrics=metrics)

	model.fit(X_train, y_train,
	          epochs=epochs,
	          verbose=0)

	y_pred = model.predict(X_test)
	report = reporter(y_test.to_numpy(), y_pred.flatten().astype(int))
	print(report)


