import numpy as np
import pandas as pd


def data_preprocessing(path):
    data = pd.read_table(path, sep='\s+', header=None, engine='python')
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values.astype('int')
    return X, y


def init(d, M, r):
    W1 = np.random.uniform(-r, r, (d, M))
    W2 = np.random.uniform(-r, r, ((M + 1), 1))
    return W1, W2


def NN(X, y, M, r, eta, T):
    N, d = X.shape
    W1, W2 = init(d, M, r)
    for i in range(T):
        pos = np.random.randint(0, N)
        X_slice = X[pos:(pos + 1), :]  # (1,d)
        y_slice = y[pos]
        # 前向传播
        s1 = np.dot(X_slice, W1)  # (1,M)
        X1 = np.tanh(s1)  # (1,M)
        X1 = np.c_[np.ones((1, 1)), X1]  # (1,M+1)
        s2 = np.dot(X1, W2)  # (1,1)
        out = np.tanh(s2)[0][0]  # float
        # 反向传播
        dout = -2 * (y_slice - out)  # (1,1)
        ds1 = dout * W2[1:].T * (1 - np.tanh(s1)**2)  # (1,M)
        dW2 = dout * X1.T  # (M,1)
        dW1 = np.dot(X_slice.T, ds1)  # (d,M)
        W2 = W2 - eta * dW2
        W1 = W1 - eta * dW1

    return W1, W2


def err(X, y, W):
    N, d = X.shape
    x = X
    for i in range(len(W) - 1):
        x = np.c_[np.ones((N, 1)), np.tanh(np.dot(x, W[i]))]
    out = np.tanh(np.dot(x, W[len(W) - 1]))
    out[out >= 0] = 1
    out[out < 0] = -1
    return np.sum(y != out) / N


if __name__ == '__main__':

    X_train, y_train = data_preprocessing('hw4_nnet_train.dat')
    X_test, y_test = data_preprocessing('hw4_nnet_test.dat')
    M = [1, 6, 11, 16, 21]
    E_out = np.zeros(5)
    for i in range(500):
        for j in range(len(M)):
            W1, W2 = NN(X_train, y_train, M[j], 0.1, 0.1, 50000)
            W = [W1, W2]
            E_out[j] += err(X_test, y_test, W)
    print(E_out / 500)

    etas = [0.001, 0.01, 0.1, 1, 10]
    E_out = np.zeros(5)
    for i in range(500):
        for j in range(len(etas)):
            W1, W2 = NN(X_train, y_train, M=3, eta=etas[j], r=0.1, T=500)
            W = [W1, W2]
            E_out[j] += err(X_test, y_test, W)
    print(E_out / 500)