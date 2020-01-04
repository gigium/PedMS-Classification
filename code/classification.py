import numpy as np
import pandas as pd

from feature_selection import read_data


from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from sklearn.metrics import classification_report, confusion_matrix



def svm(train,test,kernel='rbf'):
	y_train=train.iloc[:,-1]
	X_train=train.iloc[:,:-1]

	y_test=test.iloc[:,-1]
	X_test=test.iloc[:,:-1]



	svclassifier = SVC(kernel='rbf')
	svclassifier.fit(X_train, y_train)

	y_pred = svclassifier.predict(X_test)

	return classification_report(y_test,y_pred)




def main():
	train = read_data("./kFold/train_"+str(1)+".txt")
	test = read_data("./kFold/test_"+str(1)+".txt")

	report=svm(train,test)

	print(report)

if __name__ == '__main__':
	main()

# python classification.py




