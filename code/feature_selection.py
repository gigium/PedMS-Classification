import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SelectFromModel

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV

import warnings

import mlflow


def read_data(file_name):
	print("reading data ... reading from ", file_name)
	df = pd.read_csv(file_name, sep="\t", index_col=0)
	return df


#preliminar step
def lowMeanElimination(df, mean_tresh):
	print("\n")
	print("lowMeanElimination ... mean = ", mean_tresh)
	r, c = df.shape
	keep = (df > 0).mean() > mean_tresh
	keep['target'] = True
	cols = df.columns[keep]
	print("removed ... ", str(c-len(cols))+ " features")

	mlflow.log_param("FEATURE SELECTION-LOW MEAN mean treshold", mean_tresh)
	mlflow.log_param("FEATURE SELECTION-LOW MEAN removed features", (c-len(cols)))
	return cols
	

#preliminar step
def lowVarianceElimination(df, var_tresh):
	print("\n")
	print("lowVarianceElimination ... var = ", var_tresh)
	r, c = df.shape

	X = df.iloc[:,:-1] 
	sel_variance_threshold = VarianceThreshold(var_tresh) 
	sel_variance_threshold.fit(X)

	keep = sel_variance_threshold.get_support(indices=True)
	cols = df.columns[keep].insert(len(keep), "target")
	print("removed ... ", str(c-len(keep))+ " features")

	mlflow.log_param("FEATURE SELECTION-LOW VARIANCE variance treshold", var_tresh)
	mlflow.log_metric("FEATURE SELECTION-LOW VARIANCE removed features", (c-len(keep)))
	return cols



def correlationFElimination(df, c):
	X = df.iloc[:,:-1]  
	correlation_matrix = X.corr()
	correlated_features = set()	

	for i in range(len(correlation_matrix.columns)):
		print(i, "/", len(correlation_matrix.columns))
		for j in range(i):
			if abs(correlation_matrix.iloc[i, j]) > c:
				colname = correlation_matrix.columns[i]
				correlated_features.add(colname)

	correlated_features.add("target")
	return correlated_features



def univariateFSelect(df ,k, score_func=chi2):
	print("\n")
	print("univariateFSelect ... extracting ", str(k) + " feaures")
	print("score function ", score_func.__name__)

	X = df.iloc[:,:-1]  
	y = df.iloc[:,-1]    

	bestfeatures = SelectKBest(score_func, k)
	fit = bestfeatures.fit(X,y)

	dfscores = pd.DataFrame(fit.scores_)
	dfcolumns = pd.DataFrame(X.columns)

	dfscores = pd.DataFrame(fit.scores_)
	dfcolumns = pd.DataFrame(X.columns)
	featureScores = pd.concat([dfcolumns,dfscores],axis=1)
	featureScores.columns = ['Gene','Score']

	keep = list (featureScores.nlargest(k,'Score')['Gene'])

	mlflow.log_param("FEATURE SELECTION-UNIVARIATE score function", score_func.__name__)
	mlflow.log_param("FEATURE SELECTION-UNIVARIATE k features", k)

	keep.append("target")

	return keep



def decisionTreeFSelect(df ,k):
	print("\n")
	print("decisionTreeFSelect ... extracting ", str(k) + " feaures")

	X = df.iloc[:,:-1]  
	y = df.iloc[:,-1]    
	model = ExtraTreesClassifier()
	model.fit(X,y)
	feat_importances = pd.Series(model.feature_importances_, index=X.columns)
	
	keep = list (feat_importances.nlargest(k).index)
	keep.append("target")

	mlflow.log_param("FEATURE SELECTION-DECISION TREE k features", k)
	return keep
	# feat_importances.nlargest(30).plot(kind='barh')
	# plt.show()



# https://stats.stackexchange.com/questions/367155/why-lasso-for-feature-selection
def lassoFSelect(df, cv=5, alphas=[.1]):
	print("\n")
	print("lassoFSelect ...")

	X = df.iloc[:,:-1]  
	y = df.iloc[:,-1]    
	r,c = X.shape

	with warnings.catch_warnings():
		# ignore all caught warnings
		warnings.filterwarnings("ignore")
		selector = SelectFromModel(estimator=LassoCV(cv=cv, alphas=alphas)).fit(X, y)

	n_features = selector.transform(X).shape[1]
	print ("selected ... ", str(n_features) + " feaures" )
	 
	keep = selector.get_support(indices=True)
	cols = df.columns[keep].insert(len(keep), "target")
	print("removed ... ", str(c-len(keep))+ " features")	

	mlflow.log_param("FEATURE SELECTION-LASSO removed features", (c-len(keep)))

	return cols



def recursiveFElimination(df):
	print("\n")
	print("recursiveFElimination ... ")
	X = df.iloc[:,:-1]  
	y = df.iloc[:,-1]  	
	rfc = RandomForestClassifier(random_state=101)
	rfecv = RFECV(estimator=rfc, step=1, cv=StratifiedKFold(5), scoring='accuracy')
	rfecv.fit(X, y)
	print('Optimal number of features ... {}'.format(rfecv.n_features_))
	cols = X.columns[rfecv.support_].insert(len(X), "target")

	mlflow.log_param("FEATURE SELECTION-RECURSIVE selected features", rfecv.n_features_)

	return cols