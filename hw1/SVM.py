import numpy as np
import pandas as pd
from sklearn import svm
def data_preprocessing(path,random = False):
    data = pd.read_table(path, sep='\s+', header=None, engine='python')
    if random:
        data = data.sample(frac=1).reset_index(drop=True)
    X = data.iloc[:,1:].values
    y = data.iloc[:,0].values.astype(int)
    return X,y

def error(y_pred,y):
    return np.sum(y_pred!=y)


#15 binary classification for 0 and non-zero
'''
y_temp = y_train.copy()
y_temp = np.where(y_temp==0,1,-1)
y = y_train.copy()
clf = svm.SVC(C=0.01,kernel='linear',shrinking=False)
clf.fit(X_train,y_temp)
print(clf.coef_)
print('||w|| is '+str(np.linalg.norm(clf.coef_))) #0.57
'''
'''
16-17
n = X_train.shape[0]
for i in range(10):
    y_temp = y_train.copy()
    y_temp = np.where(y_temp==i,1,-1)
    clf = svm.SVC(C=0.01,kernel='poly',degree=2,gamma=1,coef0=1,shrinking=False)
    clf.fit(X_train,y_temp)
    y_pred = clf.predict(X_train)
    err = error(y_pred,y_temp)
    print(err/n,np.sum(abs(clf.dual_coef_)))
'''
'''
18
X_test,y_test= data_preprocessing('features_test.dat')
c = np.array([0.001,0.01,0.1,1,10])
for i in c:
    y_temp = np.where(y_train==0,1,-1)
    clf = svm.SVC(C=i,kernel='rbf',gamma=100,shrinking=False)
    clf.fit(X_train,y_temp)
    err = error(clf.predict(X_test),y_test)
    print(np.sum(clf.n_support_),err)
'''
'''
19
X_test,y_test= data_preprocessing('features_test.dat')
gammas = np.array([1,10,100,1000,10000])
for i in gammas:
    y_temp = np.where(y_train==0,1,-1)
    clf = svm.SVC(C=0.1,kernel='rbf',gamma=i,shrinking=False)
    clf.fit(X_train,y_temp)
    err = error(clf.predict(X_test),y_test)
    print(np.sum(clf.n_support_),err)
'''
test = 100
best_gamma = -1
best_err = 1.
gammas = np.array([1,10,100,1000,10000])
X_train,y_train = data_preprocessing('features_train.dat',True)
for i in range(len(gammas)):
    err = 0
    for j in range(test):
        pos = np.random.permutation(X_train.shape[0])
        X_val,y_val = X_train[pos[:1000]],y_train[pos[:1000]]
        X_train_shrinked,y_train_shrinked = X_train[pos[1000:]],y_train[pos[1000:]]
        clf = svm.SVC(C=0.1,kernel='rbf',gamma=gammas[i],shrinking=False)
        clf.fit(X_train_shrinked,y_train_shrinked)
        err += error(clf.predict(X_val),y_val)
    err/=(test*1000)
    if best_err>err:
        best_err = err
        best_gamma = gammas[i]
print(best_gamma,best_err)



