import numpy as np
import pandas as pd
import random
from decision_tree import CART

def data_preprocessing(path):
    data = pd.read_table(path, sep='\s+', header=None, engine='python')
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values.astype('int')
    return X, y

class RandomForest:
    def __init__(self, X, y, T,prune = False):
        self.X = X
        self.y = y
        self.T = T
        self.forest = self.buildForest(X, y, T,prune)

    def bootstrap(self, X, y, num):
        index = [random.randint(0, num-1) for i in range(num)]
        return X[index], y[index]

    def buildForest(self, X, y, T,prune = False):
        forest = []
        for i in range(T):
            bx, by = self.bootstrap(X, y, y.shape[0])
            forest.append(CART(bx, by,prune))
        return forest

    def fit(self, x):
        res = 0
        for i in range(self.T):
            res += self.forest[i].fit(x, self.forest[i].branch)
        return 1 if res > 0 else -1

    def predict(self,X):
        res = []
        for i in range(X.shape[0]):
            res.append(self.fit(X[i]))
        return res

def computeErr(y, y_pred):
    return np.sum(y!=y_pred)/y.shape[0]

if __name__ == '__main__':
    X_train, y_train = data_preprocessing('hw3_dectree_train.dat')
    X_test, y_test = data_preprocessing('hw3_dectree_test.dat')

    T=1
    E_in_tree = 0.
    E_in_forest = 0.
    E_out_forest = 0.
    for t in range(T):
        rf = RandomForest(X_train,y_train,300,True)
        for tree in rf.forest:
            E_in_tree += computeErr(y_train, tree.predict(X_train))
        E_in_tree/=len(rf.forest)
        E_in_forest += computeErr(y_train, rf.predict(X_train))
        E_out_forest += computeErr(y_test,rf.predict(X_test))

    E_in_tree/=T
    E_in_forest/=T
    E_out_forest/=T
    print(E_in_tree)
    print(E_in_forest,E_out_forest)


