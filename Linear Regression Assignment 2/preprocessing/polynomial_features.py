''' In this file, you will utilize two parameters degree and include_bias.
    Reference https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class PolynomialFeatures():
    
    def __init__(self, degree=2,include_bias=False):
        """
        Inputs:
        param degree : (int) max degree of polynomial features
        param include_bias : (boolean) specifies wheter to include bias term in returned feature array.
        """
        self.degree = degree
        self.include_bias = include_bias
        
        pass

    
    def transform(self,X):
        """
        Transform data to polynomial features
        Generate a new feature matrix consisting of all polynomial combinations of the features with degree less than or equal to the specified degree. 
        For example, if an input sample is  np.array([a, b]), the degree-2 polynomial features with "include_bias=True" are [1, a, b, a^2, ab, b^2].
        
        Inputs:
        param X : (np.array) Dataset to be transformed
        
        Outputs:
        returns (np.array) Tranformed dataset.
        """

        if X.ndim == 1:
            X = X.reshape(1,-1)
        D = X.shape[1]

        for deg in range(2,self.degree+1):
            for feature in range(D):
                to_add = X[:, feature].reshape(-1,1)
                X = np.hstack((X, to_add**deg))

        if self.include_bias == True:
            X = np.hstack((np.ones((X.shape[0],1)), X))

        return X

        
        pass
    
        
        
        
        
        
        
        
        
    
                
                
