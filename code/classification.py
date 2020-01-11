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
from sklearn.ensemble import VotingClassifier



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
	print("test:\t", y_test, "\npred:\t", y_pred)
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
					 lossF='categorical_crossentropy',
					 optimizerF='sgd',
					 metrics=['accuracy'],
					 epochs=40):

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
	model.add(Dense(layer2, activation=activation1)) 
	model.add(Dense(3, activation=activation2)) 

	model.compile(loss=lossF,
	              optimizer=optimizerF,
	              metrics=metrics)
	
	y_train_one_hot = np.zeros((y_train.size, y_train.max()+1))
	y_train_one_hot[np.arange(y_train.size),y_train] = 1



	model.fit(X_train, y_train_one_hot,
	          epochs=epochs,
	          verbose=0)

	y_pred = model.predict(X_test).tolist()
	cat_pred = []
	for i in range (len(y_pred)):
		cat_pred.append(np.asarray(y_pred[i]).argmax()) # integers)

	a,f,p,r = reporter(y_test.tolist(), cat_pred)
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
	a,f,p,r = reporter(y_test.tolist(),y_pred)

	return a,f,p,r



def randomForest_neuralNet_svm(train, test):
	X_train, y_train, X_test, y_test = split(train,test)

	RFclass = RandomForestClassifier(max_depth=3, random_state=0)
	RFclass.fit(X_train, y_train)
	y_pred_forest = RFclass.predict(X_test)



	SVCclass = SVC(kernel='rbf')
	SVCclass.fit(X_train, y_train)

	y_pred_svm = SVCclass.predict(X_test)



	NNet = Sequential() 
	NNet.add(Dense(32, input_dim=len(X_train.columns), activation='relu')) 
	NNet.add(Dense(32, activation='relu')) 
	NNet.add(Dense(3, activation='sigmoid')) 

	NNet.compile(loss='categorical_crossentropy',
	              optimizer='sgd',
	              metrics=['accuracy'])
	
	y_train_one_hot = np.zeros((y_train.size, y_train.max()+1))
	y_train_one_hot[np.arange(y_train.size),y_train] = 1

	NNet.fit(X_train, y_train_one_hot,
	          epochs=40,
	          verbose=0)

	y_pred = NNet.predict(X_test).tolist()
	y_pred_net = []
	for i in range (len(y_pred)):
		y_pred_net.append(np.asarray(y_pred[i]).argmax()) # integers)


	# model = VotingClassifier(estimators=[('rf', RFclass), ('svm', SVCclass), ('nnet', NNet)], voting='hard')
	# model.fit(X_train,y_train)
	# model.score(X_test,y_test)
	# majority_vote = model.predict(X_test)
	majority_vote =[]
	for i in range (len(y_pred_net)):
		# majority_vote.append(int(round((y_pred_net[i]+y_pred_svm[i]+y_pred_forest[i])/3)))
		if y_pred_net[i] == y_pred_svm[i] == y_pred_forest[i]:
			majority_vote.append(y_pred_net[i])

		# elif y_pred_svm[i]==0: 
		# 	majority_vote.append(y_pred_svm[i])

		elif y_pred_net[i] == y_pred_svm[i]:
			majority_vote.append(y_pred_net[i])

		elif y_pred_net[i] == y_pred_forest[i]:
			majority_vote.append(y_pred_net[i])

		# elif y_pred_net[i]==1: 
		# 	majority_vote.append(y_pred_net[i])

		# elif y_pred_svm[i] == y_pred_forest[i]:
		# 	majority_vote.append(y_pred_svm[i])

		# elif y_pred_forest[i]==1:
		# 	majority_vote.append(y_pred_forest[i])

		# elif y_pred_svm[i]==0:
		# 	majority_vote.append(y_pred_net[i])

		else:
			majority_vote.append(y_pred_forest[i])

	print("\nnnet\t",y_pred_net)
	print("svm\t",y_pred_svm.tolist())
	print("forest\t",y_pred_forest.tolist())
	a,f,p,r = reporter(y_test.tolist(),majority_vote)

	return a,f,p,r