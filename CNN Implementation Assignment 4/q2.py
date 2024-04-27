import numpy as np
import pandas as pd
# from logistic_regeression import LogisticRegression
# from metrics import *
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import timeit


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
N = 10
D = 10
N_test = np.arange(100,N*100+1,100)
P_test = np.arange(2,D+1,1)

time_train_n = []
time_test_n = []

for i in range(1,N+1):
    n = 100*i
    X = pd.DataFrame({i:pd.Series(np.random.randint(2, size = n)) for i in range(5)})
    y = pd.Series(np.random.randint(2, size = n))

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)
    X_train, X_test = array_to_dataframe(X_train), array_to_dataframe(X_test)
    y_train, y_test = array_to_series(y_train), array_to_series(y_test)

    start = timeit.default_timer()
    lr = LogisticRegression(random_state=0).fit(X_train,y_train)
    mid = timeit.default_timer()
    lr.predict(X_test)
    end = timeit.default_timer()

    time_train_n.append(mid-start)
    time_test_n.append(end-mid)


time_train_d = []
time_test_d = []

for d in range(2,D+1):
    
    X = pd.DataFrame({i:pd.Series(np.random.randint(2, size = 500)) for i in range(d)})
    y = pd.Series(np.random.randint(2, size = 500))

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)
    X_train, X_test = array_to_dataframe(X_train), array_to_dataframe(X_test)
    y_train, y_test = array_to_series(y_train), array_to_series(y_test)

    start = timeit.default_timer()
    lr = LogisticRegression(random_state=0).fit(X_train,y_train)
    mid = timeit.default_timer()
    lr.predict(X_test)
    end = timeit.default_timer()

    time_train_d.append(mid-start)
    time_test_d.append(end-mid)

fig1 = plt.figure()

plt.subplot(2,2,1)
plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
plt.plot(N_test, time_train_n)
plt.xlabel('Number of Samples')
plt.ylabel('Time')
plt.title('Train time with varying N')

plt.subplot(2,2,2)
plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
plt.plot(N_test, time_test_n)
plt.xlabel('Number of Samples')
plt.ylabel('Time')
plt.title('Test time with varying N')

plt.subplot(2,2,3)
plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
plt.plot(P_test, time_train_d)
plt.xlabel('Number of Features')
plt.ylabel('Time')
plt.title('Train time with varying D')

plt.subplot(2,2,4)
plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
plt.plot(P_test, time_test_d)
plt.xlabel('Number of Features')
plt.ylabel('Time')
plt.title('Test time with varying D')

plt.suptitle('Train and test time in Logistic Regression')
plt.savefig('q2.png')
plt.show()