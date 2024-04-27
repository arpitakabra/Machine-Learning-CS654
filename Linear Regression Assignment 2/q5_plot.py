import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from preprocessing.polynomial_features import PolynomialFeatures
from linearRegression.linearRegression import LinearRegression
from sklearn.linear_model import SGDRegressor

x = np.array([i*np.pi/180 for i in range(60,300,4)])
np.random.seed(10)  #Setting seed for reproducibility
y = 4*x + 7 + np.random.normal(0,3,len(x))

if x.ndim == 1:
    x = x.reshape(-1,1)

y = pd.Series(y)
x_copy = pd.DataFrame(x)
theta = []

LR = SGDRegressor()
fit_sgd = LR.fit(x, y) # here you can use fit_non_vectorised / fit_autograd methods
theta.append(np.sqrt(np.sum(np.square(LR.coef_))))

deg_vary = 5
for deg in range(2,deg_vary+1):
    poly = PolynomialFeatures(deg)
    X = poly.transform(x)
    x_copy = pd.DataFrame(X)
    LR.fit(x_copy, y)
    theta.append(np.sqrt(np.sum(np.square(LR.coef_))))

degrees = np.array(range(1,deg_vary+1))

fig1 = plt.plot()
plt.scatter(degrees, theta)

plt.title('Theta vs Poly degree')
plt.xlabel('Polynomial degree')
plt.ylabel('Value of theta corresponding to each feature')
plt.savefig('q5_plot.png')
plt.show()





