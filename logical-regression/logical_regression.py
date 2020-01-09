import numpy as np

def sigmoid(x):
    return 1./1.+np.exp(-x)

def init_with_zeros(dim):
    # calculate weights and bias together
    w = np.zeros((dim, 1))
    b = 0.
    return w, b

def forward_propagation(W, b, X, Y):
    Z = np.dot(W, X)+b
    A = sigmoid(Z)
    # the number of sample
    m = X.shape[1]
    cost = -1./m*np.sum(Y*np.log(A)+(1-Y)*np.log(1-A))
    return A, cost

# lr: learning rate
def backforward_propagation(W, b, X, Y, A, lr):
    m = X.shape[1]
    dZ = Y-A
    # calculate gradient
    dW = 1./m*np.dot(X, dZ.T)
    db = 1./m*np.sum(dZ)

    # updata w, b
    W = W-dW*lr
    b = b-db*lr

    return W, b

def train(W, b, X, Y, num_iter, lr):
    for it in range(num_iter):
        A, cost = forward_propagation(W, b, X, Y)
        backforward_propagation(W, b, X, Y, A, lr)
    return W, b

