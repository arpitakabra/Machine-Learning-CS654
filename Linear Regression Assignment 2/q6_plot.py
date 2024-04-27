from cProfile import label
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from preprocessing.polynomial_features import PolynomialFeatures
from linearRegression.linearRegression import LinearRegression
from sklearn.linear_model import SGDRegressor
# from matplotlib.pyplot import cm 

LR = SGDRegressor()
deg_vary = 3
degrees = np.array(range(1,deg_vary+1))

fig1 = plt.plot()

for n in range(300, 1201, 200):

    x = np.array([i*np.pi/180 for i in range(60,n,4)])
    np.random.seed(10)  #Setting seed for reproducibility
    y = 4*x + 7 + np.random.normal(0,3,len(x))

    if x.ndim == 1:
        x = x.reshape(-1,1)
    
    y = pd.Series(y)
    x_copy = pd.DataFrame(x)
    theta = []

    fit_sgd = LR.fit(x, y) 
    theta.append(np.sqrt(np.sum(np.square(LR.coef_))))

    for deg in range(2,deg_vary+1):
        poly = PolynomialFeatures(deg)
        X = poly.transform(x)
        x_copy = pd.DataFrame(X)
        LR.fit(x_copy, y)
        theta.append(np.sqrt(np.sum(np.square(LR.coef_))))
    
    plt.plot(degrees, theta, label=n)

plt.title('Theta Vs Degree for varying N')
plt.legend()
plt.xlabel('Varying degree')
plt.ylabel('Theta')
plt.savefig('q6_plot.png')
plt.show()
    

