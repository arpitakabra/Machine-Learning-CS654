# import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from numpy import exp
# from metrics import *
import autograd.numpy as np
from autograd import grad

# Import Autograd modules here


class Multi_LogisticRegression():
    def __init__(self):
        '''
        :param fit_intercept: Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations (i.e. data is expected to be centered).
        '''
        self.coef_ = None  # Replace with numpy array or pandas series of coefficients learned using using the fit methods

        pass


    def fit_vectorised(self, X, y,lamda =0, n_iter=100, lr=0.01):

        n_samples = X.shape[0]
        n_features =X.shape[1]
        # find number of targets
        n_targets = np.unique(y).size
        self.coef_ = np.ones((n_features,n_targets))
        grad_ = 0
        alpha = 0
        # X =X.to_numpy()
        # y=y.to_numpy()
        for i in range(1, n_iter+1):
            alpha = lr
            coeff = self.coef_
            cost = grad(self.loss_function, 0)
            grad_ = cost(coeff, X, y, lamda,n_targets,n_samples)

            coeff = coeff - alpha*grad_
            self.coef_ = coeff.copy()

        pass


    def predict(self, X):
        pred = X.dot(self.coef_)
        # print("pred",pred)
        prob =self.softmax(pred)
        # print("prob",prob)
        pred_class = prob.argmax(axis =1)
        print(pred_class)
        pred_class= pd.Series(pred_class.flatten())
        return(pred_class)

    def loss_function(self, theta, X, y, lamda,n_targets,n_samples):
        pred = (X@theta)
        softmax =self.find_softmax(pred,y,n_targets,n_samples)
        y_ = np.eye(n_targets)[y]
        y_=y_.reshape(n_samples,n_targets)
        # print(((np.log(softmax))@y_.T).shape)
        loss = np.trace((np.log(softmax))@y_.T) + lamda*np.sum(np.sum(theta**2))
        loss = loss/n_samples
        # print(loss)
        # print(loss.shape)
        return(loss)

    def find_softmax(self,pred,y,n_targets,n_samples):
        exp_pred = np.exp(pred)
        row_sum = exp_pred.sum(axis=1)
        softmax = exp_pred / row_sum[:,np.newaxis]
        # den = np.sum(np.exp(pred),axis =1)
        # print(den.shape)
        # y =np.eye(n_targets)[y]
        # y=y.reshape(n_samples,n_targets)
        # num =pred@y.T
        # num =np.exp(np.diagonal(num))
        # softmax = num/den
        return softmax

    def softmax(self,pred):
        soft =((np.exp(pred)).T/np.sum(np.exp(pred),axis =1)).T
        return soft

    def plot_surface(self, X, y, t_0, t_1):
        pass
