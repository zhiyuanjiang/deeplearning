import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model
import matplotlib

# implementation nn by myself
# this is a simple neural network of three layers
# addr:https://github.com/dennybritz/nn-from-scratch/blob/master/nn-from-scratch.ipynb
""" record
at the begin, my derivative_function is:
def derivative_function(x):
    return np.maximum(0, 1)
my activation function become a linear function, so by training the model, my fitting function always is linear.

last, by change the number of node and training, (parameters)
the conclude is great
"""

# change default figure size
matplotlib.rcParams['figure.figsize'] = (10.0, 8.0)

# generate a dataset and plot it
# use make_moons function
np.random.seed(0)
X, y = sklearn.datasets.make_moons(200, noise=0.2)
# plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)
# plt.show()

"""
# Train the logistic regression classifier
clf = sklearn.linear_model.LogisticRegressionCV()
clf.fit(X, y)
"""

# Helper function to plot a decision boundary.
# If you don't fully understand this function don't worry, it just generates the contour plot below.
def plot_decision_boundary(pred_func):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)

"""
# Plot the decision boundary
plot_decision_boundary(lambda x: clf.predict(x))
plt.title("Logistic Regression")
plt.show()
"""

# three layers neural network
def init_weights(n1, n2, n3):
    W1 = np.random.randn(n2, n1)
    W2 = np.random.randn(n3, n2)
    b1 = np.zeros((n2, 1))
    b2 = np.zeros((n3, 1))
    params = {
        'W1':W1,'W2':W2,'b1':b1,'b2':b2
    }
    return params

def ReLu(x):
    return np.maximum(0, x)

def derivative_ReLu(x):
    return np.where(x > 0, 1., 0.)

def loss_function(a, y):
    return -y*np.log(a)-(1-y)*np.log(1-a)

def sigmoid(x):
    return 1./(1.+np.exp(-x))

def forward_propagation(X, y, params, m):
    Z1 = np.dot(params['W1'], X)+params['b1']
    A1 = ReLu(Z1)
    Z2 = np.dot(params['W2'], A1)+params['b2']
    A2 = sigmoid(Z2)
    loss = 1./m*np.sum(loss_function(A2, y), axis=1)
    #print(loss)
    store_var = {
        'Z1':Z1, 'A1':A1, 'Z2':Z2, 'A2':A2
    }
    return params, store_var, loss

def backward_propagation(X, y, params, var, m, lr):
    dZ2 = var['A2']-y
    dW2 = 1./m*np.dot(dZ2, var['A1'].T)
    db2 = 1./m*np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.dot(params['W2'].T, dZ2)*derivative_ReLu(var['Z1'])
    dW1 = 1./m*np.dot(dZ1, X.T)
    db1 = 1./m*np.sum(dZ1, axis=1, keepdims=True)

    params['W1'] -= lr*dW1
    params['b1'] -= lr*db1
    params['W2'] -= lr*dW2
    params['b2'] -= lr*db2

    return params

def train(X, y, params, iter_num, m, lr):
    re = []
    a = 0
    for i in range(iter_num):
        params, store_var, loss = forward_propagation(X, y, params, m)
        params = backward_propagation(X, y, params, store_var, m, lr)
        re.append(loss)
        a += loss
        if i%1000 == 0:
            print('the loss is %f'%(loss/100.))
            ans = predict(params, X.T)
            b = 0
            a = 0
            for j in range(ans.shape[1]):
                if ans[0,j] == y[j]:
                    b = b+1
            print('the precise is %f'%(b/ans.shape[1]))
    return params,re

def predict(params, x):
    x = x.T
    Z1 = np.dot(params['W1'], x) + params['b1']
    A1 = ReLu(Z1)
    Z2 = np.dot(params['W2'], A1) + params['b2']
    A2 = sigmoid(Z2)
    return np.where(A2 > 0.5, 1., 0.)

X = X.T
attr_num = X.shape[0]
m = X.shape[1]
params = init_weights(attr_num, 5, 1)
params, re = train(X, y, params, 20000, m, 0.1)

X = X.T
plot_decision_boundary(lambda x:predict(params, x))
plt.show()

#print(predict(params, np.array([[0,0.12]])))

# a = [i for i in range(len(re))]
# plt.plot(a, re, color='red', linewidth=2)
# plt.show()