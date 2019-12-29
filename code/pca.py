import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def _PCA(df, targets, COMPONENTS=47):
    
    #mlflow.log_param("COMPONENTS",COMPONENTS)# log of parameters


    X = pd.DataFrame(StandardScaler().fit_transform(df))
    X.head()
    pca = PCA(n_components=COMPONENTS)
    principalComponents = pca.fit_transform(X)

    columns = []
    for i in range(COMPONENTS):
        columns.append("PC"+str(i+1))

    principalDf = pd.DataFrame(data = principalComponents, columns=columns)
    targets = pd.DataFrame(targets)
    targets = targets.rename(columns={0:"targets"})
    finalDf = pd.concat([principalDf, targets], axis = 1)

    variance=np.round(pca.explained_variance_ratio_* 100, decimals =2)[:20]
    print(variance)
    