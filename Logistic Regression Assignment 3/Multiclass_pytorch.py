import torch
import torch.nn as nn
from sklearn import datasets
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
iris = datasets.load_iris()


class logreg_multicls_torch(nn.Module):
    def __init__(self, input_dim, num_cls, fit_intercept=True):
        super(logreg_multicls_torch, self).__init__()
        self.fit_intercept = fit_intercept
        self.linear = nn.Linear(input_dim, num_cls, bias=self.fit_intercept)
        self.softmax = nn.Softmax()

    def forward(self, X):
        z = self.linear(X)
        return(z)

    def fit(self, X, y, fit_intercept=True, batch_size=2, n_iters=100, lr=0.01, lamda=0):
        '''
            Function to train model using logistic regression. Autograd is used

            param model: model on which to train
            param X: pd.DataFrame with rows as samples and coulumns as features
            param y: pd.Series with rows as outputs
            param batch_size: int specifying the batch size. It lies between 1 and total samples in data
            param n_iter: number of iterations in training
            param lr: learning rate
            param lamda: lambda for applying L2 penalty
            '''
        N = X.shape[0]
        D = X.shape[1]
        optimizer = torch.optim.SGD(self.parameters(), lr=lr)

        for itr in range(n_iters):
            optimizer.zero_grad()
            indices = [i % N for i in range(
                itr*batch_size, (itr+1)*batch_size)]
            X_batch = torch.tensor(X[indices], dtype=torch.float)
            y_batch = torch.tensor(y[indices])
            y_pred = self.forward(X_batch)
            loss = reg_CEloss(self.parameters(), y_batch, y_pred, lamda)
            loss.backward()
            optimizer.step()


def reg_CEloss(params, y, y_pred, lamda):
    l2 = 0
    y = y.squeeze()
    y_pred = y_pred.squeeze()
    for param in params:
        l2 += param.square().sum()
    criterion = nn.CrossEntropyLoss()

    return(criterion(y_pred, y)+lamda*l2)


data = pd.DataFrame(iris.data)
target = pd.DataFrame(iris.target)

data = data.to_numpy()
target = target.to_numpy()
n_split = 5
skf = StratifiedKFold(n_splits=n_split)
for train_index, test_index in skf.split(data, target):

    X_train, X_test = data[train_index], data[test_index]
    y_train, y_test = target[train_index], target[test_index]
    LR = logreg_multicls_torch(4, 3)
    LR.fit(X_train, y_train)
    y_prob_test = model(torch.tensor(X_test, dtype=torch.float))
    y_test_cls = (y_prob_test > 0.5).squeeze().numpy()
    print(f'Test accuracy: {np.sum(y_test_cls == y_test)/len(y_test):.2f}')
