import numpy as np
import pandas as pd


def data_preprocessing(path):
    data = pd.read_table(path,sep='\s+',header=None)
    X = data.iloc[:,:2].values
    y = data.iloc[:,-1].values
    return X,y

def gaussian_kernel(X,X_hat,gamma):
    kernel_matrix = np.zeros((X.shape[0], X_hat.shape[0]))
    for i in range(X_hat.shape[0]):
        kernel_matrix[:,i] = np.sum((X-X_hat[i])**2, 1)
    kernel_matrix = np.exp(-gamma*kernel_matrix)
    return kernel_matrix

if __name__ == '__main__':
    X,y = data_preprocessing('hw2_lssvm_all.dat')
    X_train,y_train = X[:400,:],y[:400]
    X_test,y_test = X[400:,:],y[400:]

    gamma = [32, 2, 0.125]
    lamb = [0.001, 1, 1000]
    Ein = np.zeros((len(gamma), len(lamb)))
    Eout = np.zeros((len(gamma), len(lamb)))
    for i in range(len(gamma)):
        K = gaussian_kernel(X_train, X_train, gamma[i])
        K2 = gaussian_kernel(X_train, X_test, gamma[i])
        for j in range(len(lamb)):
            beta = np.linalg.inv(lamb[j] * np.eye(X_train.shape[0]) + K).dot(y_train)
            y_in_pred = np.sign(K.dot(beta))
            Ein[i, j] = np.sum(y_in_pred != y_train) / X_train.shape[0]
            y_out_pred = np.sign(K2.T.dot(beta))
            Eout[i, j] = np.sum(y_out_pred != y_test) / X_test.shape[0]
    print(np.min(Ein))
    print(np.min(Eout))