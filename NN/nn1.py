import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model
import matplotlib

#############################
# build a neural network of four layers
#
#############################

# Display plots inline and change default figure size
matplotlib.rcParams['figure.figsize'] = (10.0, 8.0)

# Generate a dataset and plot it
np.random.seed(0)
X, y = sklearn.datasets.make_moons(200, noise=0.20)

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


def init_weights(n0, n1, n2):
    W1 = np.random.randn(n1, n0)
    b1 = np.zeros((n1, 1))
    W2 = np.random.randn(n2, n1)
    b2 = np.zeros((n2, 1))
    W3 = np.random.randn(2, n2)
    b3 = np.zeros((2, 1))
    params = {
        'W1':W1,
        'b1':b1,
        'W2':W2,
        'b2':b2,
        'W3':W3,
        'b3':b3
    }
    return params

def ReLu(x):
    return np.maximum(0, x)

def derivative_ReLu(x):
    return np.where(x > 0, 1., 0.)

def build_model(test_x, test_y, params, iter_num, lr, print_loss=True):
    W1,W2,W3,b1,b2,b3 = params['W1'],params['W2'],params['W3'],params['b1'],params['b2'],params['b3']
    x,y = test_x,test_y
    m = len(test_y)
    for i in range(iter_num):
        # forward propagation
        z1 = np.dot(W1, x)+b1
        a1 = ReLu(z1)
        z2 = np.dot(W2, a1)+b2
        a2 = ReLu(z2)
        z3 = np.dot(W3, a2)+b3

        # calculation loss
        exp_scores = np.exp(z3)
        # print(exp_scores)
        probs = exp_scores/np.sum(exp_scores, axis=0)
        loss = -np.log(probs[y, range(m)])
        loss_sum = 1./m*np.sum(loss)
        if print_loss and i%1000 == 0:
            print(loss_sum)

        # backward propagation
        dz3 = probs
        dz3[y, range(m)] -= 1
        dw3 = 1./m*np.dot(dz3, a2.T)
        db3 = 1./m*np.sum(dz3, axis=1, keepdims=True)
        dz2 = np.dot(W3.T, dz3)*derivative_ReLu(z2)
        dw2 = 1./m*np.dot(dz2, a1.T)
        db2 = 1./m*np.sum(dz2, axis=1, keepdims=True)
        dz1 = np.dot(dw2.T, dz2)*derivative_ReLu(z1)
        dw1 = 1./m*np.dot(dz1, x.T)
        db1 = 1./m*np.sum(dz1, axis=1, keepdims=True)

        W1 -= lr*dw1
        W2 -= lr*dw2
        W3 -= lr*dw3
        b1 -= lr*db1
        b2 -= lr*db2
        b3 -= lr*db3

def predict(model, x):
    x = x.T
    W1, b1, W2, b2, W3, b3 = model['W1'], model['b1'], model['W2'], model['b2'], model['W3'], model['b3']
    # Forward propagation
    z1 = np.dot(W1, x)+b1
    a1 = ReLu(z1)
    z2 = np.dot(W2, a1)+b2
    a2 = ReLu(z2)
    z3 = np.dot(W3, a2)+b3
    exp_scores = np.exp(z3)
    probs = exp_scores / np.sum(exp_scores, axis=0, keepdims=True)
    # print(probs.shape)
    return np.argmax(probs, axis=0)

attr_num = X.shape[1]
m = X.shape[0]
params = init_weights(attr_num, 11, 11)
build_model(X.T, y, params, 20000, 0.1)
plot_decision_boundary(lambda x:predict(params,x))

plt.show()

