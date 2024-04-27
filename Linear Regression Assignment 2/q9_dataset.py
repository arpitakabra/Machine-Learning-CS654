import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linearRegression.linearRegression import LinearRegression
from metrics import *

np.random.seed(42)

N = 200
P = 3
X = np.random.randn(N, 3)
y = pd.Series(np.random.randn(N))

LR = LinearRegression(fit_intercept=True)
LR.fit_vectorised(pd.DataFrame(X), y, batch_size=1) # here you can use fit_non_vectorised / fit_autograd methods
y_hat1 = LR.predict(pd.DataFrame(X))

print(LR.return_theta())
print('RMSE: ', rmse(y_hat1, y))
print('MAE: ', mae(y_hat1, y))

X_mcol = np.hstack((X, (X[:,0]*10 + X[:,1]).reshape(-1,1)))
X_mcol = pd.DataFrame(X_mcol)

LR.fit_vectorised(X_mcol, y, batch_size=1)
y_hat2 = LR.predict(X_mcol)
print(LR.return_theta())
print('RMSE: ', rmse(y_hat2, y))
print('MAE: ', mae(y_hat2, y))


