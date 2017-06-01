from load import load
from sklearn import decomposition
import sys

ndim = int(sys.argv[1])

X = load('../data/train.txt')
Y = load('../data/train_label.txt').flatten()
X_test = load('../data/test.txt')
Y_real_test = load('../data/test_label.txt').flatten()


# PCA
pca = decomposition.PCA(n_components=ndim)
pca.fit(X)
X_red = pca.transform(X)
X_red_test = pca.transform(X_test)
