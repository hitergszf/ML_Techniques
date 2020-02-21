import numpy as np
import pandas as pd


def data_preprocessing(path):
    data = pd.read_table(path, sep='\s+', header=None, engine='python')
    X = data.iloc[:, :].values
    return X


def K_Means_Cluster(X, k=1):
    # initialize
    index = np.random.randint(0, X.shape[0] - 1, size=k)
    centers = X[index]
    clusters = [[] for i in range(k)]
    while (True):
        old_clusters = clusters[:]
        clusters = [[] for i in range(k)]
        # Step1:find nearest center to fit
        for i in range(X.shape[0]):
            dis = [distance(X[i], centers[j]) for j in range(k)]
            label = np.argsort(dis)[0]
            clusters[label].append(X[i])

        # Step2:update the mean with the average
        for j in range(k):
            if len(clusters[j]) > 0:
                x = np.zeros_like(clusters[j][0])
                for i in range(len(clusters[j])):
                    x = x + clusters[j][i]
                centers[j] = x / len(clusters[j])
        # Check the condition
        flag = True
        for i in range(k):
            if len(old_clusters[i]) > 0:
                if old_clusters[i] != clusters[i]:
                    flag = False
                    break
        if flag:
            break
    return np.array(clusters), np.array(centers)


def fit(x, clusters):
    for i in range(len(clusters)):
        if len(clusters[i]) > 0:
            for j in range(len(clusters[i])):
                if (x == clusters[i][j]).all():
                    return i
    return -1


def predict(X, clusters):
    N = X.shape[0]
    categories = [fit(X[i], clusters) for i in range(N)]
    return categories


def distance(x, mu):
    return np.sum((x - mu) ** 2)


def err(X, centers, clusters):
    N = X.shape[0]
    k = centers.shape[0]
    error = 0.
    for i in range(N):
        for j in range(k):
            if len(clusters[j]) > 0:
                for t in range(len(clusters[j])):
                    if (X[i] == clusters[j][t]).all():
                        error += distance(X[i], centers[j])
                break
    error /= N
    return error

if __name__ == '__main__':
    X = data_preprocessing('hw4_kmeans_train.dat')
    T = 500
    Err1 = 0.
    Err2 = 0.
    for  T in range(T):
        clusters, centers = K_Means_Cluster(X, k=2)
        Err1+=err(X,centers,clusters)
        clusters, centers = K_Means_Cluster(X, k=10)
        Err2 += err(X, centers, clusters)
    print(Err1/T)
    print(Err2/T)
