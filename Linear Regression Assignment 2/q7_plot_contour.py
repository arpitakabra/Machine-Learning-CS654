import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linearRegression.linearRegression import LinearRegression
import numpy as np
from matplotlib.animation import FuncAnimation
from IPython import display
import time
from matplotlib.widgets import Slider, Button
from matplotlib import animation,rc
from IPython.display import HTML, Image

# Figure = plt.figure()
N = 30
P = 1
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series(np.random.randn(N))

LR = LinearRegression()
n_iter=20

out=LR.fit_vectorised(X, y,batch_size=N,n_iter=n_iter,plotting=True) # here you can use fit_non_vectorised / fit_autograd methods
thetas=out[1]
y_hat=out[0]
print(thetas.shape)
print(y_hat.shape)

Figure = plt.figure()
plt.yticks(np.arange(-1, 1, 0.1))


def AnimationFunction(frame):

    if frame!=n_iter-1:
        plt.clf()
     
    plt.ylim((-1,1))
    plt.scatter(np.array(X),y)
    plt.plot(np.array(X),y_hat[frame])
    plt.text(0.8, 1.1, "Theta0="+str(round(thetas[frame][0],3))+" Theta1="+str(round(thetas[frame][1],3)), bbox=dict(facecolor='bisque', alpha=0.5))
    plt.suptitle("Iteration=" +str(frame))
    plt.pause(0.1)
   
   

anim_created = animation.FuncAnimation(Figure, AnimationFunction, frames=n_iter, interval=25,repeat=False)
# anim_created.save('q7_gif.gif', fps=30)
plt.show()
plt.close()

##### Creating a slider
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.35)

    
plt.scatter(np.array(X), y)
l, = plt.plot(np.array(X), y_hat[0])

axfreq = plt.axes([0.25, 0.15, 0.65, 0.03])


# plt.show()
iter = Slider(axfreq, 'Iterations', 1,n_iter-1, 1,valstep=1)
def update(val):
    l.set_ydata(y_hat[val])
 
# Call update function when slider value is changed
iter.on_changed(update)

 
# display graph
plt.show()