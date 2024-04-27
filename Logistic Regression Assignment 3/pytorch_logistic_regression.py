from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt


class logistic_reg_torch(nn.Module):
    def __init__(self, input_dim, fit_intercept=True):
        super(logistic_reg_torch, self).__init__()
        self.fit_intercept = fit_intercept
        self.linear = nn.Linear(input_dim, 1, bias=self.fit_intercept)
        self.sigmoid = nn.Sigmoid()
        self.lamda = 0

    def forward(self, X):
        z = self.linear(X)
        y_prob = self.sigmoid(z)
        return(y_prob)

    def fit_torch(self, X, y, fit_intercept=True, batch_size=2, n_iters=100, lr=0.01, lamda=0):
        '''
            Function to train model using logistic regression with pytorch

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
        loss_iter = []
        self.lamda = lamda

        for itr in range(n_iters):
            optimizer.zero_grad()
            indices = [i % N for i in range(
                itr*batch_size, (itr+1)*batch_size)]
            X_batch = torch.tensor(X[indices], dtype=torch.float)
            y_batch = torch.tensor(y[indices], dtype=torch.float)
            y_pred = self.forward(X_batch)
            loss = reg_BCEloss(self.parameters(), y_batch, y_pred,  lamda)
            loss.backward()
            loss_iter.append(np.float64(loss))
            optimizer.step()
            print(f'Iteration {itr}: Loss {loss:.2f}')
        return(loss_iter)

    def plot_surface(self, X, y):

        xx = np.linspace(-3, 3, 50)
        yy = np.linspace(-3, 3, 50)

        XX, YY = np.meshgrid(xx, yy)

        plt.figure(figsize=(10, 5))

        Z = self.forward(torch.tensor(
            np.c_[XX.ravel(), YY.ravel()], dtype=torch.float))
        Z = Z.detach().numpy()
        Z = Z.reshape(XX.shape)

        im = plt.imshow(Z, interpolation='nearest', extent=(XX.min(), XX.max(
        ), YY.min(), YY.max()), aspect='auto', origin='lower', cmap=plt.cm.PuOr_r)
        contour = plt.contour(
            XX, YY, Z, levels=[0.5], linewidths=2, colors=['k'])
        plt.scatter(X[:, 0], X[:, 1], s=30, c=y,
                    cmap=plt.cm.Paired, edgecolors=(0, 0, 0))
        plt.xticks(())
        plt.yticks(())
        plt.axis([-3, 3, -3, 3])
        plt.colorbar(im)
        plt.title('Dataset prediction probabilities')
        plt.xlabel('Featrue x1')
        plt.ylabel('Feature x2')
        plt.grid()

        plt.tight_layout()
        if self.lamda == 0:
            plt.savefig('q2_decision_surface.png')
        else:
            plt.savefig('q2_decision_surface_l.png')
        plt.show()


def reg_BCEloss(params, y, y_pred, lamda):
    l2 = 0
    y = y.squeeze()
    y_pred = y_pred.squeeze()
    for param in params:
        l2 += param.square().sum()
    criterion = nn.BCELoss(reduction='mean')

    return(criterion(y_pred, y)+lamda*l2)


rng = np.random.RandomState(0)
X = rng.randn(200, 2)
y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0)

# Random split of dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=0)
model = logistic_reg_torch(2)
loss_itr1 = model.fit_torch(X_train, y_train, lr=0.001, n_iters=100, lamda=0)
model.plot_surface(X_test, y_test)
# Train and test accuracy
y_prob_test = model(torch.tensor(X_test, dtype=torch.float))
y_test_cls = (y_prob_test > 0.5).squeeze().numpy()
print(f'Test accuracy: {np.sum(y_test_cls == y_test)/len(y_test):.2f}')

y_prob_train = model(torch.tensor(X_train, dtype=torch.float))
y_train_cls = (y_prob_train > 0.5).squeeze().numpy()
print(f'Train accuracy: {np.sum(y_train_cls == y_train)/len(y_train):.2f}')



loss_itr2 = model.fit_torch(X_train, y_train, lr=0.001, n_iters=100, lamda=0.5)
model.plot_surface(X_test, y_test)
# Train and test accuracy
y_prob_test = model(torch.tensor(X_test, dtype=torch.float))
y_test_cls = (y_prob_test > 0.5).squeeze().numpy()
print(f'Test accuracy: {np.sum(y_test_cls == y_test)/len(y_test):.2f}')

y_prob_train = model(torch.tensor(X_train, dtype=torch.float))
y_train_cls = (y_prob_train > 0.5).squeeze().numpy()
print(f'Train accuracy: {np.sum(y_train_cls == y_train)/len(y_train):.2f}')


plt.plot(loss_itr1)
plt.plot(loss_itr2)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.savefig('q2_loss_comp.png')
plt.title('Loss Comparison')
plt.legend(['lambda=0', 'lambda=0.5'])
plt.show()

# Draw the decision surface




