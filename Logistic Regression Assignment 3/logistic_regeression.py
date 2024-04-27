# import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Import Autograd modules here
from autograd import grad
import autograd.numpy as np
from sklearn.model_selection import PredefinedSplit


class LogisticRegression():
    def __init__(self):
        '''
        :param fit_intercept: Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations (i.e. data is expected to be centered).
        '''
        self.coef_ = None #Replace with numpy array or pandas series of coefficients learned using using the fit methods
        self.loss_iter = []
        self.lamda = 0



    def fit_regression(self, X, y, batch_size=1, n_iter=200, lr=0.01, lamda = 0):
        '''
        Function to train model using logistic regression. Autograd is used

        param X: pd.DataFrame with rows as samples and coulumns as features
        param y: pd.Series with rows as outputs
        param batch_size: int specifying the batch size. It lies between 1 and total samples in data
        param n_iter: number of iterations in training
        param lr: learning rate
        param lamda: lambda for applying L2 penalty
        '''

        N = len(X)
        D = len(X.columns)

        X_copy = np.hstack((np.ones((N,1)), X))
        features = D+1
        
        theta = np.zeros((features,1))

        start_idx = 0
        end_idx = start_idx + batch_size
        loss = []

        for iter in range(n_iter):

            if end_idx > N:
                index_list = list(range(0,end_idx-len(X)))
                index_list.extend(range(start_idx, len(X)))
                start_idx = len(X) - start_idx
                end_idx = start_idx + batch_size
            else:
                index_list = list(range(start_idx, end_idx))
                start_idx = end_idx
                end_idx += batch_size
            
            X_train = X_copy[index_list]
            y_train = y.to_numpy()[index_list]
            y_train = y_train.reshape(-1,1)

            cost = grad(self.ce_loss, 0)
            cost_iter = cost(theta, X_train, y_train, lamda)/batch_size
            loss.append(self.ce_loss(theta, X_train, y_train, lamda))

            theta = theta - lr*cost_iter
        
        theta = theta.reshape(-1,1)
        self.coef_ = theta
        self.loss_iter = loss
        self.lamda = lamda
            
        pass



    def decision_prob(self, X):
        '''
        Funtion to run the LinearRegression on a data point

        :param X: pd.DataFrame with rows as samples and columns as features

        :return: y: pd.Series with rows corresponding to output variable. The output variable in a row is the prediction for sample in corresponding row in X.
        '''
        theta = self.coef_.reshape(-1,1)
        n = len(X)
        X_test = np.hstack((np.ones((n,1)),X))
        y_pred = 1/(1 + np.exp(-X_test@theta))
        y_pred = pd.Series(y_pred.flatten())

        return y_pred
    
    def predict(self, X):

        y_pred = self.decision_prob(X)
        y_pred = np.round(y_pred)

        return y_pred

        pass

    def plot_surface(self, X, y):
        
        xx = np.linspace(-3,3,50)
        yy = np.linspace(-3,3,50)

        XX,YY = np.meshgrid(xx,yy)

        plt.figure(figsize=(10,5))

        Z = self.decision_prob(np.c_[XX.ravel(),YY.ravel()])
        Z = Z.to_numpy()
        Z = Z.reshape(XX.shape)

        im = plt.imshow(Z, interpolation='nearest', extent=(XX.min(), XX.max(), YY.min(), YY.max()), aspect='auto', origin='lower',cmap=plt.cm.PuOr_r)
        contour = plt.contour(XX, YY, Z, levels=[0.5], linewidths=2, colors=['k'])
        plt.scatter(X[0], X[1], s = 30, c=y, cmap=plt.cm.Paired, edgecolors=(0,0,0))
        plt.xticks(())
        plt.yticks(())
        plt.axis([-3, 3, -3, 3])
        plt.colorbar(im)
        plt.title('Dataset prediction probabilities')
        plt.xlabel('Featrue x1')
        plt.ylabel('Feature x2')
        plt.grid()

        plt.tight_layout()
        if self.lamda==0:
            plt.savefig('lr_lamda_0.png')
        else:
            plt.savefig('lr_lamda.png')
        plt.show()

        pass



    def ce_loss(self, theta, X, y, lamda):
        pred = (X@theta)
        sigmoid = 1/(1+np.exp(-pred))
        loss = -np.sum(y*np.log(sigmoid) + (1-y)*np.log(1 - sigmoid)) + lamda*theta.T@theta

        return loss
    

    def return_theta(self):
        return self.coef_
