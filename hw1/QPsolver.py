from sklearn import svm
import numpy as np

x = np.array([[1, 0], [0, 1], [0, -1], [-1, 0], [0, 2], [0, -2], [-2, 0]])
y = [-1, -1, -1, 1, 1, 1, 1]

clf = svm.SVC(C=1e100, kernel='poly', degree=2, coef0=1,gamma=1)
clf.fit(x, y)

alpha = clf._dual_coef_
sv = clf.support_vectors_
b = clf.intercept_
print(alpha)
print(sv)
print(b)