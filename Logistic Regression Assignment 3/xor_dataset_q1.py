import numpy as np
import pandas as pd
from logistic_regeression import LogisticRegression
from metrics import *
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


###################################################################

#function to convert dataset to dataframe if not
def array_to_dataframe(x):
    x = pd.DataFrame(x)
    x = x.reset_index(drop=True)
    return x

#function to convert labels to series if not already
def array_to_series(x):
    x = pd.Series(x)
    x = x.reset_index(drop= True)
    return x

###################################################################


xx, yy = np.meshgrid(np.linspace(-3,3,50), np.linspace(-3,3, 50))

range = np.random.RandomState(0)
X = range.randn(500, 2)
Y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0)

X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.4)
X_train, X_test = array_to_dataframe(X_train), array_to_dataframe(X_test)
y_train, y_test = array_to_series(y_train), array_to_series(y_test)

LG = LogisticRegression()

######## with lambda = 0 #########
LG.fit_regression(X_train, y_train, batch_size=1)
y_hat_train = LG.predict(X_train)
y_hat_test = LG.predict(X_test)
fig = LG.plot_surface(X_train, y_train)
print('Train accuracy: ', accuracy(y_hat_train, y_train))
print('Test accuracy: ', accuracy(y_hat_test, y_test))

loss_0 = np.array(LG.loss_iter)

######## with lambda = 0.5 #########
LG.fit_regression(X_train, y_train, batch_size=1, lamda=0.5)
y_hat_train = LG.predict(X_train)
y_hat_test = LG.predict(X_test)
fig = LG.plot_surface(X_train, y_train)
print('Train accuracy: ', accuracy(y_hat_train, y_train))
print('Test accuracy: ', accuracy(y_hat_test, y_test))

loss_1 = np.array(LG.loss_iter)

##### plots comparing loss ######

n_iter = len(loss_0)
loss_0 = loss_0.reshape(n_iter,1)

n = np.arange(1,n_iter+1,1)
loss_1 = loss_1.reshape(n_iter,1)

plt.plot(n,loss_0)
plt.plot(n,loss_1)
plt.title('Loss Comparison with lambda as 0 and 0.5')
plt.xlabel('Iteration number')
plt.ylabel('Loss')
plt.legend(['lambda=0', 'lambda=0.5'])
plt.savefig('loss_comparison.png')
plt.show()