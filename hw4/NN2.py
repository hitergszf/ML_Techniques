import numpy as np
import pandas as pd


def data_preprocessing(path):
    data = pd.read_table(path, sep='\s+', header=None, engine='python')
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values.astype('int')
    return X, y


def init(d0, d1, d2, r):
    W1 = np.random.uniform(-r, r, (d0, d1))
    W2 = np.random.uniform(-r, r, (d1 + 1, d2))
    W3 = np.random.uniform(-r, r, (d2 + 1, 1))
    return W1, W2, W3


def NN(X, y, d1, d2, r, eta, T):
    N, d = X.shape
    W1, W2, W3 = init(d, d1, d2, r)
    for i in range(T):
        pos = np.random.randint(0, N)
        X_slice = X[pos:(pos + 1), :]  # (1,d)
        y_slice = y[pos]
        # 前向传播
        s1 = np.dot(X_slice, W1)  # (1,d1)
        X1 = np.tanh(s1)  # (1,d1)
        X1 = np.c_[np.ones((1, 1)), X1]  # (1,d1+1)
        s2 = np.dot(X1, W2)  # (1,d2)
        X2 = np.tanh(s2)  # (1,d2)
        X2 = np.c_[np.ones((1, 1)), X2]  # (1,d2+1)
        s3 = np.dot(X2, W3)
        out = np.tanh(s3)[0][0]
        # 反向传播
        dout = -2 * (y_slice - out)  # (1,1)
        ds2 = dout * W3[1:].T * (1 - np.tanh(s2) ** 2)  # (1,d2)
        ds1 = np.dot(ds2, W2[1:].T) * (1 - np.tanh(s1) ** 2)  # (1,d1)
        dW3 = dout * X2.T  # (d2+1,1)
        dW2 = ds2 * X1.T  # (d1+1,1)
        dW1 = np.dot(X_slice.T, ds1)  # (d0,d1)
        W3 = W3 - eta * dW3
        W2 = W2 - eta * dW2
        W1 = W1 - eta * dW1

    return W1, W2, W3


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

    E_out = 0
    for i in range(50):
        W1, W2, W3 = NN(X_train, y_train, 8, 3, eta=0.01, r=0.1, T=50000)
        W = [W1, W2, W3]
        E_out += err(X_test, y_test, W)
    print(E_out / 50)
