import numpy as np
import pandas as pd
import random

class CART:
    def __init__(self, X, y,prune=False):
        self.X = X
        self.y = y
        self.internode = 0
        self.branch = self.buildTree(X, y,prune)

    def buildTree(self, X, y,prune=False):
        branches = []
        #one layer
        if prune:
            branch = self.learningBranch(X, y)
            X1, y1, X2, y2 = self.split_Dataset(X, y, branch)
            branches.append(branch)
            if y1.shape[0]>0:
                branches.append([1])
            else:
                branches.append([-1])
            if y2.shape[0]>0:
                branches.append([-1])
            else:
                branches.append([1])
            return branches
        #fully grown
        if abs(np.sum(y)) == y.shape[0] or np.sum(X!=X[0,:])==0:
            branches.append(y[np.argmax(np.argsort(y))])
            return branches
        else:
            branch = self.learningBranch(X, y)
            X1, y1, X2, y2 = self.split_Dataset(X, y, branch)
            branches.append(branch)
            branches.append(self.buildTree(X1, y1))
            branches.append(self.buildTree(X2, y2))
            self.internode = self.internode + 1
            return branches
    def learningBranch(self, X, y):
        N = X.shape[1]
        branch = {}
        branch['sign'] = 0
        branch['theta'] = 0
        branch['dim'] = 0
        best_score = 100

        for s in [-1, 1]:
            for d in range(N):
                X_sort = np.sort(X[:, d])
                thetas = []
                thetas.append(X_sort[0] - 0.1)
                for j in range(1, X_sort.shape[0]):
                    thetas.append((X_sort[j] + X_sort[j - 1]) / 2)
                for j in range(len(thetas)):
                    hypothesis = {'sign': s, 'theta': thetas[j], 'dim': d}
                    if self.impurity_score(X, y, hypothesis) < best_score:
                        best_score = self.impurity_score(X, y, hypothesis)
                        branch = hypothesis
        return branch

    def split_Dataset(self, X, y, hypothesis):
        sign, dim, theta = hypothesis['sign'], hypothesis['dim'], hypothesis['theta']
        y_pred = sign * np.sign(X[:, dim] - theta)
        positive_X, positive_y = X[y_pred == 1], y[y_pred == 1]
        negative_X, negative_y = X[y_pred == -1], y[y_pred == -1]
        print(positive_y.shape[0],negative_y.shape[0])
        return positive_X, positive_y, negative_X, negative_y

    def gini_impurity(self, y):
        if y.shape[0] == 0:
            return 1
        positive = (np.sum(y == 1) / y.shape[0])
        negative = (np.sum(y == -1) / y.shape[0])
        return 1 - positive ** 2 - negative ** 2

    def impurity_score(self, X, y, hypothesis):
        X1, y1, X2, y2 = self.split_Dataset(X, y, hypothesis)
        return y1.shape[0] * self.gini_impurity(y1) + y2.shape[0] * self.gini_impurity(y2)

    def fit(self, x, branches):
        if len(branches) == 3:
            current_hypothesis = branches[0]
            dim, sign, theta = current_hypothesis['dim'], current_hypothesis['sign'], current_hypothesis['theta']
            y_pred = sign * np.sign(x[dim] - theta)
            if y_pred == 1:
                return self.fit(x, branches[1])
            else:
                return self.fit(x, branches[2])
        return branches[0]

    def predict(self, X):
        result = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            x = X[i]
            result[i] = self.fit(x, branches=self.branch)
        return result

    def getInternelNode(self):
        return self.internode

def data_preprocessing(path):
    data = pd.read_table(path, sep='\s+', header=None, engine='python')
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values.astype('int')
    return X, y
def computeErr(y, y_pred):
    return np.sum(y!=y_pred)/y.shape[0]

if __name__ == '__main__':
    X_train, y_train = data_preprocessing('hw3_dectree_train.dat')
    X_test, y_test = data_preprocessing('hw3_dectree_test.dat')
    tree = CART(X_train,y_train)
    print(tree.getInternelNode())
    print(computeErr(y_train,tree.predict(X_train)))
    print(computeErr(y_test,tree.predict(X_test)))