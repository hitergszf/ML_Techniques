import numpy as np
import pandas as pd

def data_preprocessing(path):
    data = pd.read_table(path, sep='\s+', header=None, engine='python')
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values.astype('int')
    return X, y

def distance(x1,x2):
    return np.sum((x1-x2)**2)

def fit(x,X_train,y_train,k=1):
    dis = [distance(x,X_train[i]) for i in range(X_train.shape[0])]
    index = np.argsort(dis)[0:k]
    y = np.sign(np.sum(y_train[index]))
    return y

def predict(X,X_train,y_train,k):
    y_pred = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        y_pred[i] = fit(X[i],X_train,y_train,k)

    return y_pred


def err(y,y_pred):
    return np.sum(y!=y_pred)/y.shape[0]

if __name__ == '__main__':
    X_train,y_train = data_preprocessing('hw4_knn_train.dat')
    X_test,y_test = data_preprocessing('hw4_knn_test.dat')
    E_in_1 = err(y_train,predict(X_train,X_train,y_train,1))
    E_out_1 = err(y_test,predict(X_test,X_train,y_train,1))
    print(E_in_1,E_out_1)
    E_in_5 = err(y_train, predict(X_train, X_train, y_train, 5))
    E_out_5 = err(y_test, predict(X_test, X_train, y_train, 5))
    print(E_in_5, E_out_5)