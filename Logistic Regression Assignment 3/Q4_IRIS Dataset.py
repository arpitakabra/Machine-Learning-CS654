from sklearn import datasets
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
from Logistic_Regression import Multi_LogisticRegression
iris = datasets.load_iris()
from metrices import *

data =pd.DataFrame(iris.data)
target =pd.DataFrame(iris.target)

data =data.to_numpy()
target =target.to_numpy()
n_split =5
skf = StratifiedKFold(n_splits =n_split)
for train_index, test_index in skf.split(data, target):
    X_train, X_test = data[train_index], data[test_index]
    y_train, y_test = target[train_index], target[test_index]
    lambda_ =[0,0.5]
    for lam in lambda_:
        LR = Multi_LogisticRegression()
        LR.fit_vectorised(X_train, y_train,lam)
        y_hat_test = LR.predict(X_test)
        y_hat_train =LR.predict(X_train)
        print(y_train)
        print("predicted",y_hat_train)
        # # y_hat = pd.Series(y_hat.flatten())
        print('Test accuracy: ', accuracy(y_hat_test, y_test))
        print('Train accuracy: ', accuracy(y_hat_train, y_train))
    break


        

