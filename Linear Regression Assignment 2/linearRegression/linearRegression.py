import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Import Autograd modules here
from autograd import grad
from sklearn.model_selection import PredefinedSplit

class LinearRegression():
    def __init__(self, fit_intercept=True):
        '''
        :param fit_intercept: Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations (i.e. data is expected to be centered).
        '''
        self.fit_intercept = fit_intercept
        self.coef_ = None #Replace with numpy array or pandas series of coefficients learned using using the fit methods

      

    def fit_non_vectorised(self, X, y, batch_size=1, n_iter=100, lr=0.01, lr_type='constant'):
        '''
        Function to train model using non-vectorised gradient descent.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param batch_size: int specifying the batch size. Batch size can only be between 1 and number of samples in data. 
        :param n_iter: number of iterations (default: 100)
        :param lr: learning rate (default: 0.01)
        :param lr_type: If lr_type = 'constant', then the learning rate remains constant,
                        if lr_type = 'inverse', then learning rate = lr / t, where t = current iteration number

        :return None
        '''
        N = len(X)
        D = len(X.columns)

        if self.fit_intercept == True:
            X_copy = np.hstack((np.ones((N,1)), X))
            features = D + 1
        else:
            X_copy = X.copy().to_numpy()
            features = D

        theta = np.zeros(features)

        start_idx = 0
        end_idx = start_idx + batch_size

        for iter in range(n_iter):

            if lr_type != 'constant':
                lr = lr/iter

            if end_idx > len(X):
                 index_list = list(range(0,end_idx-len(X)))
                 index_list.extend(range(start_idx, len(X)))
                 start_idx = len(X) - start_idx
                 end_idx = start_idx + batch_size
            else:
                index_list = list(range(start_idx, end_idx))
                start_idx = end_idx
                end_idx += batch_size
            
            X_train = pd.DataFrame(X_copy[index_list])
            y_train = pd.Series(y.to_numpy()[index_list])

            y_pred = []
            cost = []
            for index, row in X_train.iterrows():
                y_predi = 0
                for param in range(features):
                    
                    y_predi += row[param]*theta[param]
                
                y_pred.append(y_predi)

            for i in range(features):

                for index, row in X_train.iterrows():

                    cost.append(-2*(y_train[index] - y_pred[index])*row[i])
                cost[i] = cost[i]/len(y_train)

            for i in range(features):

                theta[i] -= lr*cost[i]

        self.coef_ = theta



    def fit_vectorised(self, X, y,batch_size, n_iter=100, lr=0.01, lr_type='constant', plotting=False):
        '''
        Function to train model using vectorised gradient descent.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param batch_size: int specifying the batch size. Batch size can only be between 1 and number of samples in data.
        :param n_iter: number of iterations (default: 100)
        :param lr: learning rate (default: 0.01)
        :param lr_type: If lr_type = 'constant', then the learning rate remains constant,
                        if lr_type = 'inverse', then learning rate = lr / t, where t = current iteration number

        :return None
        '''

        X=np.array(X)
        y=np.array(y)
        N= X.shape[0]
        M= X.shape[1]

        if(self.fit_intercept):   
            X_dash= np.hstack((np.ones((N,1)),X))
            thetas= np.zeros(M+1)

        else:
            X_dash=X.copy()
            thetas=np.zeros(M)

        thetas=thetas.reshape(-1,1)
        y_dash=y.reshape(-1,1)
        
     
        start_index=0
        end_index=batch_size
        y_per_iteration=np.empty((n_iter,N))
        thetas_per_iteration=np.empty((n_iter,2))
        for i in range(n_iter):
            if(end_index>N):
                end_index-=N
                X_batch=np.concatenate((X_dash[start_index:N,:],X_dash[:end_index,:]))
                y_batch=np.concatenate((y_dash[start_index:N,:],y_dash[:end_index,:]))
            else:
                X_batch=X_dash[start_index:end_index,:]
                y_batch=y_dash[start_index:end_index,:]

            if lr_type != 'constant':
                lr = lr/i

            y_pred=X_batch@thetas
            errors=np.mean(np.square(y_pred-y_batch))
            delta_mse=((2*(y_pred-y_batch).T@X_batch).T)/y_batch.size
            thetas=thetas-lr*delta_mse
            start_index=end_index
            end_index=start_index+batch_size

            if(plotting==True):
                temp=thetas.reshape((thetas.shape[0]))
                y_p=X_dash@temp
                y_per_iteration[i]=y_p.reshape((y_p.shape[0]))
                thetas_per_iteration[i]=temp

        self.coef_=thetas.reshape((thetas.shape[0]))
        if(plotting==True):
            return [y_per_iteration,thetas_per_iteration]



    def fit_autograd(self, X, y, batch_size=1, n_iter=100, lr=0.01, lr_type='constant'):
        '''
        Function to train model using gradient descent with Autograd to compute the gradients.
        Autograd reference: https://github.com/HIPS/autograd

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param batch_size: int specifying the  batch size. Batch size can only be between 1 and number of samples in data.
        :param n_iter: number of iterations (default: 100)
        :param lr: learning rate (default: 0.01)
        :param lr_type: If lr_type = 'constant', then the learning rate remains constant,
                        if lr_type = 'inverse', then learning rate = lr / t, where t = current iteration number

        :return None
        '''
        N = len(X)
        D = len(X.columns)

        if self.fit_intercept == True:
            X_copy = np.hstack((np.ones((N,1)), X))
            features = D + 1
        else:
            X_copy = X.copy().to_numpy()
            features = D
        
        theta = np.zeros(features)
        start_idx = 0
        end_idx = start_idx + batch_size

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

            cost = grad(self.autogradient, 0)
            cost_iter = cost(theta, X_train, y_train)

            theta = theta - lr*cost_iter
        
        theta = theta.reshape(-1,1)
        self.coef_ = theta
            

        pass



    def fit_normal(self, X, y):
        '''
        Function to train model using the normal equation method.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))

        :return None
        '''

        N= X.shape[0]
        M= X.shape[1]
        X_dash= np.hstack((np.ones((N,1)),X))
        thetas= np.random.randn(M+1)
        thetas=thetas.reshape(-1,1)
        y_dash=y.to_numpy().reshape(-1,1)

        theta = np.linalg.inv(X_dash.T@X_dash)@(X_dash.T@y_dash)

        self.coef_ = theta

        pass

    def predict(self, X):
        '''
        Funtion to run the LinearRegression on a data point

        :param X: pd.DataFrame with rows as samples and columns as features

        :return: y: pd.Series with rows corresponding to output variable. The output variable in a row is the prediction for sample in corresponding row in X.
        '''
        theta = self.coef_.reshape(-1,1)
        n = len(X)
        if self.fit_intercept == True:
            X_test = np.hstack((np.ones((n,1)),X))
        else:
            X_test = X.copy()
        y_pred = X_test@theta
        y_pred = pd.Series(y_pred.flatten())

        return y_pred

        pass

    def plot_surface(self, X, y, t_0, t_1):
        """
        Function to plot RSS (residual sum of squares) in 3D. A surface plot is obtained by varying
        theta_0 and theta_1 over a range. Indicates the RSS based on given value of t_0 and t_1 by a
        red dot. Uses self.coef_ to calculate RSS. Plot must indicate error as the title.

        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param t_0: Value of theta_0 for which to indicate RSS
        :param t_1: Value of theta_1 for which to indicate RSS

        :return matplotlib figure plotting RSS
        """

        pass

    def plot_line_fit(self, X, y, t_0, t_1):
        """
        Function to plot fit of the line (y vs. X plot) based on chosen value of t_0, t_1. Plot must
        indicate t_0 and t_1 as the title.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param t_0: Value of theta_0 for which to plot the fit
        :param t_1: Value of theta_1 for which to plot the fit

        :return matplotlib figure plotting line fit
        """
        

        pass

    def plot_contour(self, X, y, t_0, t_1):
        """
        Plots the RSS as a contour plot. A contour plot is obtained by varying
        theta_0 and theta_1 over a range. Indicates the RSS based on given value of t_0 and t_1, and the
        direction of gradient steps. Uses self.coef_ to calculate RSS.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param t_0: Value of theta_0 for which to plot the fit
        :param t_1: Value of theta_1 for which to plot the fit

        :return matplotlib figure plotting the contour
        """

        pass


    def autogradient(self, theta, X, y):

        y_pred = X@theta
        error = y - y_pred
        
        return (1/len(X))*(error.T@error)

    def return_theta(self):
        return self.coef_

