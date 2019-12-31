import sys

import pandas as pd
from sklearn.feature_selection import VarianceThreshold

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LassoCV

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

import mlflow
import mlflow.sklearn



def read_data(file_name):
	df = pd.read_csv(file_name, sep="\t", index_col=0)
	return df



def lowMeanElimination(df, mean_tresh):
	keep = (df > 0).mean() > mean_tresh
	return df[df.columns[keep]]
	


def lowVarianceElimination(df, var_tresh):
	X = df.iloc[:,:-1] 
	sel_variance_threshold = VarianceThreshold(var_tresh) 
	return pd.DataFrame(sel_variance_threshold.fit_transform(X))



def univariateFSelect(df ,k, score_func=chi2):
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
	
	return df[keep]



def decisionTreeFSelect(df ,k):
	X = df.iloc[:,:-1]  
	y = df.iloc[:,-1]    
	model = ExtraTreesClassifier()
	model.fit(X,y)
	feat_importances = pd.Series(model.feature_importances_, index=X.columns)
	
	keep = list (feat_importances.nlargest(k).index)

	return df[keep]
	# feat_importances.nlargest(30).plot(kind='barh')
	# plt.show()


#returns only features with a score greater than 0
# it can be done selecting the first k features
def lassoFSelect(df):
	X = df.iloc[:,:-1]  
	y = df.iloc[:,-1]    
	reg = LassoCV()
	reg.fit(X, y)
	coef = pd.Series(reg.coef_, index = X.columns)	
	keep = []
	indexes = list(coef.index)

	for i in range(0, len(indexes)):
		if coef[i] > 0:
			keep.append(indexes[i])
	
	return df[keep]


def main():
	data = read_data(sys.argv[1])
	#TODO

 
# python feature_selection.py .\data\new_data.txt
if __name__== "__main__":
  main()
