import numpy as np
import pandas as pd

from feature_selection import read_data

from keras.models import Sequential
from keras.layers import Dense

from sklearn import svm
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (accuracy_score, f1_score, precision_score, 
								recall_score, classification_report, 
								confusion_matrix)

import mlflow

import warnings

from collections import Counter

from sklearn.ensemble import RandomForestClassifier



def split(train,test):
	y_train=train.iloc[:,-1]
	X_train=train.iloc[:,:-1]
	y_test=test.iloc[:,-1]
	X_test=test.iloc[:,:-1]
	print("\n")
	print("train classes ", sorted(Counter(y_train).items()))
	print("test classes ", sorted(Counter(y_test).items()))

	# mlflow.log_param("test classes", sorted(Counter(y_test).items()))

	return X_train, y_train, X_test, y_test



def reporter(y_test,y_pred):
	print("_____________________________________________________________________\n")	
	print("test: ", y_test, "\npred: ", y_pred)
	# mlflow.log_param("test", y_test)
	# mlflow.log_param("pred", y_pred)
	print("_____________________________________________________________________\n")
	with warnings.catch_warnings():
		# ignore all caught warnings
		warnings.filterwarnings("ignore")
		print(classification_report(y_test,y_pred))	
		print(confusion_matrix(y_test, y_pred))
		a = accuracy_score(y_test, y_pred)
		f = f1_score(y_test, y_pred, average="macro")
		p = precision_score(y_test, y_pred, average="macro")
		r = recall_score(y_test, y_pred, average="macro")

	return a,f,p,r
	


def svm(train,test,kernel='rbf'):
	print("\n")
	print("svm ... ")
	print("kernel function : ", kernel)

	mlflow.log_param("CLASSIFICATION-SVM kernel function", kernel)
	
	X_train, y_train, X_test, y_test = split(train,test)

	svclassifier = SVC(kernel='rbf')
	svclassifier.fit(X_train, y_train)

	y_pred = svclassifier.predict(X_test)
	a,f,p,r = reporter(y_test.to_numpy(),y_pred)

	return a,f,p,r



def decision_tree(train,test,criterion):
	print("\n")
	print("decision_tree ... ")

	X_train, y_train, X_test, y_test = split(train,test)

	classifier = DecisionTreeClassifier(criterion=criterion)
	classifier.fit(X_train, y_train)	

	y_pred = classifier.predict(X_test)

	a,f,p,r = reporter(y_test,y_pred)
	return a,f,p,r



def feedForwardNN(train, test,
					 layer1=32, layer2=32, 
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
	print("CLASSIFICATION-NN layers", [layer1, layer2])
	print("CLASSIFICATION-NN optimizer function", optimizerF)
	print("CLASSIFICATION-NN epochs", epochs)

	mlflow.log_param("CLASSIFICATION-NN loss function", lossF)
	mlflow.log_param("CLASSIFICATION-NN optimizer function", optimizerF)
	mlflow.log_param("CLASSIFICATION-NN layers", [layer1, layer2])
	mlflow.log_param("CLASSIFICATION-NN optimizer function", optimizerF)
	mlflow.log_param("CLASSIFICATION-NN epochs", epochs)

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
	a,f,p,r = reporter(y_test.to_numpy(), y_pred.flatten().astype(int))
	return a, f, p, r



def randomForest(train, test, max_depth=3, random_state=0):
	print("\n")
	print("randomForest ... ")
	print("max_depth : ", max_depth)

	mlflow.log_param("CLASSIFICATION-randomForest max_depth", max_depth)

	X_train, y_train, X_test, y_test = split(train,test)

	classifier = RandomForestClassifier(max_depth=3, random_state=0)
	classifier.fit(X_train, y_train)
	
	y_pred = classifier.predict(X_test)
	a,f,p,r = reporter(y_test.to_numpy(),y_pred)

	return a,f,p,r



