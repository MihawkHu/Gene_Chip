from load import load
from sklearn import decomposition
from sklearn.neighbors import KNeighborsClassifier
import sys

# python pca-knn.py ndim
 
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

# knn
knn = KNeighborsClassifier()
knn = knn.fit(X_red, Y)
Y_test = knn.predict(X_red_test)

cnt, tot = 0, len(Y_test)
for i in range(len(Y_test)):
    if Y_test[i] == Y_real_test[i]:
        cnt += 1

print('knn: pca %d' % (ndim))
print('accuracy: %f' % (cnt/tot))
