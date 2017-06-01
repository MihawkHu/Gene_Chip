from load import load
from sklearn import decomposition
from lr import logistic_regression
import sys

# python pca-lr.py ndim reg

ndim = int(sys.argv[1])
regularizer = sys.argv[2]

X = load('../data/train.txt')
Y = load('../data/train_label.txt').flatten()
X_test = load('../data/test.txt')
Y_real_test = load('../data/test_label.txt').flatten()


# PCA
pca = decomposition.PCA(n_components=ndim)
pca.fit(X)
X_red = pca.transform(X)
X_red_test = pca.transform(X_test)

# LR
Y_test = logistic_regression(X_red, Y, regularizer, X_red_test)

cnt, tot = 0, len(Y_test)
for i in range(len(Y_test)):
    if Y_test[i] == Y_real_test[i]:
        cnt += 1

print('logreg: pca %d, regularizer %s' % (ndim, regularizer))
print('accuracy: %f' % (cnt/tot))
