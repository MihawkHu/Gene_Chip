from sklearn.svm import SVC
# from sklearn import datasets
# import numpy as np

# iris = datasets.load_iris()
# X = iris.data
# Y = iris.target
# test_X = np.array([[100, 2, 3, 100]])

def svm(X, Y, test_X):
    svc = SVC()
    svc.fit(X, Y)
    return svc.predict(test_X)