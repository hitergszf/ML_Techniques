import numpy as np
import pandas as pd

def data_preprocessing(path):
    data = pd.read_table(path,sep='\s+',header=None)
    X = data.iloc[:,:2].values
    y = data.iloc[:,-1].values
    return X,y


def decision_stump(X,y,U):
    N,M = X.shape[0],X.shape[1]
    thetas = np.zeros(N)
    best_E_in = 1.
    best_s = 0
    best_theta = -1
    best_dim = -1
    best_pred = np.zeros((y.shape[0],1))
    for dim in range(M):
        temp = np.sort(X[:,dim])
        thetas[0] = X[0,dim]-0.1
        for i in range(1,N):
            thetas[i] = (temp[i-1]+temp[i])/2
        for s in [-1,1]:
            for j in range(N):
                y_pred = s*np.sign(temp-thetas[j])
                err = (y_pred!=y)
                err = err*np.ones(err.shape[0]).astype(float)
                err = err.reshape(-1,1)
                weighted_err = np.sum(U*err)
                E_in = weighted_err/np.sum(U)
                #print(E_in)
                if E_in < best_E_in:
                    best_E_in = E_in
                    best_dim = dim
                    best_s = s
                    best_theta = thetas[j]
                    best_pred = y_pred
                #print(dim,s,thetas[j],E_in)
    return best_dim,best_theta,best_s,best_E_in,best_pred


def adaboost(X,y,T):
    U = np.ones(X.shape[0],dtype='float32')/X.shape[0]
    U = U.reshape(-1,1)
    parameters = []
    for i in range(T):
        dim,theta,sign,E_in,pred = decision_stump(X,y,U)
        scaling_factor = np.sqrt((1-E_in)/E_in)
        #print(err.shape,U.shape)
        U[pred!=y] *=scaling_factor
        U[pred==y] /=scaling_factor
        #print(U)
        alpha = np.log(scaling_factor)
        parameter = {}
        parameter['dim'] = dim
        parameter['sign'] = sign
        parameter['theta'] = theta
        parameter['alpha'] = alpha
        parameters.append(parameter)
    return parameters,U

def predict(parameters,X):
    res = 0.
    for i in range(len(parameters)):
        dim = parameters[i]['dim']
        sign = parameters[i]['sign']
        theta = parameters[i]['theta']
        alpha = parameters[i]['alpha']
        res += alpha*sign*np.sign(X[:,dim]-theta)
    return np.sign(res).astype(int)

def compute_err(y,y_pred):
    return np.mean(y!=y_pred)

if __name__ == '__main__':
    X_train,y_train = data_preprocessing('hw2_adaboost_train.dat') # (100,2) (100,)
    X_test, y_test = data_preprocessing('hw2_adaboost_test.dat')

    parameters,U = adaboost(X_train, y_train, 1)
    #res = predict(parameters, X_train)
    #print(compute_err(y_train, res))
    #print(U.sum())


    res = predict(parameters, X_test)
    print(compute_err(y_test, res))

    parameters, U = adaboost(X_train, y_train, 300)
    res = predict(parameters, X_test)
    print(compute_err(y_test, res))


