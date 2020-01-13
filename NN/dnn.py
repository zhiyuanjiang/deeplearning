# Package imports
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model
import matplotlib

####################
# mmp, there is a problem, but I can't find problem.
#
####################

# Display plots inline and change default figure size
# %matplotlib inline
matplotlib.rcParams['figure.figsize'] = (10.0, 8.0)

# Generate a dataset and plot it
np.random.seed(0)
X, y = sklearn.datasets.make_moons(200, noise=0.20)

def plot_decision_boundary(pred_func):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    fuck = np.c_[xx.ravel(), yy.ravel()]
    Z = pred_func(fuck)
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)


def init_weights(attr_num, layers):
    m = len(layers)
    params = {}
    for i in range(m):
        if i == 0:
            params["W"+str(i+1)] = np.random.randn(layers[0], attr_num)
            params["b"+str(i+1)] = np.random.randn(layers[0], 1)
        else:
            params["W"+str(i+1)] = np.random.randn(layers[i], layers[i-1])
            params["b"+str(i+1)] = np.random.randn(layers[i], 1)
    return params

def ReLu(x):
    return np.maximum(0, x)

def Sigmoid(x):
    return 1./(1.+np.exp(-x))

def activate_func(data, func="ReLu"):
    if func == "ReLu":
        return ReLu(data)
    elif func == "tanh":
        return np.tanh(data)

def forward_propagation(X, layers_num, params, func):
    a = X
    cache = {}
    for i in range(layers_num):
        z = np.dot(params["W" + str(i + 1)], a) + params["b" + str(i + 1)]
        if i != layers_num-1:
            a = activate_func(z, func)
            cache["z"+str(i+1)] = z
            cache["a"+str(i+1)] = a
        else:
            a = Sigmoid(z)
            cache["z"+str(i+1)] = z
            cache["a"+str(i+1)] = a
    cache["a0"] = X
    return cache

def loss_func(a, y):
    return -y*np.log(a)-(1-y)*np.log(1-a)

def derivative_func(data, func):
    if func == "ReLu":
        return np.where(data > 0, 1., 0.)
    elif func == "tanh":
        return 1.-np.power(data, 2)

def backward_propagation(cache, layers_num, y, func):
    dz = cache["a"+str(layers_num)]-y
    m = len(y)
    grads = {}
    grads["dz"+str(layers_num)] = dz
    for i in range(layers_num):
        id = layers_num-i
        if id != layers_num:
            dz = np.dot(grads["dw"+str(id+1)].T, grads["dz"+str(id+1)])*derivative_func(cache["z"+str(id)], func)
            grads["dz"+str(id)] = dz

        dw = 1./m*np.dot(dz, cache["a"+str(id-1)].T)
        db = 1./m*np.sum(dz, axis=1, keepdims=True)
        grads["dw"+str(id)] = dw
        grads["db"+str(id)] = db

    return grads


def build_model(X, y, layers_num, params, func, iter_num, lr):
    m = len(y)

    for i in range(iter_num):

        cache = forward_propagation(X, layers_num, params, func)

        loss = loss_func(cache['a'+str(layers_num)], y)
        loss_sum = 1./m*np.sum(loss)
        if i % 1000 == 0:
            print("the loss is : "+str(loss_sum))

        grads = backward_propagation(cache, layers_num, y, func)

        for j in range(layers_num):
            params["W"+str(j+1)] -= lr*grads["dw"+str(j+1)]
            params["b"+str(j+1)] -= lr*grads["db"+str(j+1)]

    print("end")

def predict(x, layers_num, params, func):
    a = x.T
    for i in range(layers_num):
        z = np.dot(params["W" + str(i + 1)], a) + params["b" + str(i + 1)]
        if i != layers_num - 1:
            a = activate_func(z, func)
        else:
            a = Sigmoid(z)
    return np.where(a > 0.5, 1, 0)

# the number of neural of last layer always is one
layers = [23,1]
attr_num = X.shape[1]
func = "ReLu"
params = init_weights(attr_num, layers)
build_model(X.T, y, len(layers), params,  func, 20000, 0.1)
plot_decision_boundary(lambda x:predict(x, len(layers), params, func))
plt.show()