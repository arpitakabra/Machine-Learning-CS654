from cProfile import label
from imaplib import Time2Internaldate
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linearRegression.linearRegression import LinearRegression
import timeit

LR = LinearRegression(fit_intercept=True)


############# Varying sample size #################


gd_time = []
normal_time = []
N = 2100
x_vec = np.linspace(30, N, int((N-30)/30))
for n in range(30, N, 30):

    X = pd.DataFrame(np.random.randn(n, 1))
    y = pd.Series(np.random.randn(n))
    start_time = timeit.default_timer()
    LR.fit_vectorised(X, y, batch_size=n) # here you can use fit_non_vectorised / fit_autograd methods
    mid_time = timeit.default_timer()
    LR.fit_normal(X, y)
    end_time = timeit.default_timer()

    gd_time.append(mid_time-start_time)
    normal_time.append(end_time-mid_time)

fig1 = plt.plot()
plt.subplot(1,3,1)
plt.tight_layout(h_pad=2, w_pad=2, pad=2.5)
plt.plot(x_vec, gd_time, label='Gradient Descent')
plt.plot(x_vec, normal_time, label='Normal Method')
plt.xlabel('Sample Size (N)')
plt.ylabel('Time')
plt.legend()
plt.title('Combined plots')

plt.subplot(1,3,2)
plt.plot(x_vec, gd_time)
plt.xlabel('Sample Size (N)')
plt.ylabel('Time')
plt.title('Gradient Descent')

plt.subplot(1,3,3)
plt.plot(x_vec, normal_time)
plt.xlabel('Sample Size (N)')
plt.ylabel('Time')
plt.title('Normal Method')

plt.suptitle('Time plots for varying N')
plt.savefig('q8_varying_N.png')
plt.show()


#############Varying number of features#################

gd_time = []
normal_time = []
D = 40
x_vec = np.linspace(1, D, int(D/2))
for d in range(1, D, 2):

    X = pd.DataFrame(np.random.randn(100, d))
    y = pd.Series(np.random.randn(100))
    start_time = timeit.default_timer()
    LR.fit_vectorised(X, y, batch_size=100) # here you can use fit_non_vectorised / fit_autograd methods
    mid_time = timeit.default_timer()
    LR.fit_normal(X, y)
    end_time = timeit.default_timer()

    gd_time.append(mid_time-start_time)
    normal_time.append(end_time-mid_time)

fig2 = plt.plot()
plt.subplot(1,3,1)
plt.tight_layout(h_pad=2, w_pad=2, pad=2.5)
plt.plot(x_vec, gd_time, label='Gradient Descent')
plt.plot(x_vec, normal_time, label='Normal Method')
plt.xlabel('Number of features (D)')
plt.ylabel('Time')
plt.legend()
plt.title('Combined plots')

plt.subplot(1,3,2)
plt.plot(x_vec, gd_time)
plt.xlabel('Number of features (D)')
plt.ylabel('Time')
plt.title('Gradient Descent')

plt.subplot(1,3,3)
plt.plot(x_vec, normal_time)
plt.xlabel('Number of features (D)')
plt.ylabel('Time')
plt.title('Normal Method')

plt.suptitle('Time plots for varying features')
plt.savefig('q8_varying_D.png')
plt.show()