import sys
import numpy as np
import mlflow
import mlflow.sklearn

from feature_selection import lowVarianceElimination, correlationFElimination, read_data
from oversampling import randomOverSampling, SMOTEOverSampling

from collections import Counter



from sklearn import  linear_model
from sklearn.metrics import mean_squared_error, r2_score


def main():
	
	train = read_data("./kFold/train_"+str(1)+".txt")
	test = read_data("./kFold/test_"+str(1)+".txt")
	train_oversampled = SMOTEOverSampling(train, 3)
	trainFinal = lowVarianceElimination(train_oversampled, .9)

	print(sorted(Counter(trainFinal["target"]).items()))

	trainX = trainFinal.iloc[:, :-1]
	trainy = trainFinal.iloc[:, -1]

	testX = test.iloc[:, :-1]
	testy = test.iloc[:, -1]
	print(trainX.shape)
	print(trainy.shape)

	regr = linear_model.RidgeClassifier()

	# Train the model using the training sets
	regr.fit(trainX.to_numpy(), trainy)

	# Make predictions using the testing set
	diabetes_y_pred = regr.predict(testX.to_numpy())

	print(diabetes_y_pred)
	print(testy)

	# The mean squared error
	print('Mean squared error: %.2f'
	      % mean_squared_error(testy, diabetes_y_pred))
	# The coefficient of determination: 1 is perfect prediction
	print('Coefficient of determination: %.2f'
	      % r2_score(testy, diabetes_y_pred))




# after the run of the script, lunch 'mlflow ui' command 
# and go 'to http://localhost:5000' to see the ui
# python find_best_features.py .\data\new_data.txt
if __name__== "__main__":
	main()
