from load import load
from sklearn import decomposition
from sklearn.svm import SVC
import sys

# python pca-svm.py ndim kernel
# kernel can be linear, poly, rbf, sigmoid
 
ndim = int(sys.argv[1])
kern = sys.argv[2]

X = load('../data/train.txt')
Y = load('../data/train_label.txt').flatten()
X_test = load('../data/test.txt')
Y_real_test = load('../data/test_label.txt').flatten()


# PCA
pca = decomposition.PCA(n_components=ndim)
pca.fit(X)
X_red = pca.transform(X)
X_red_test = pca.transform(X_test)

# SVM 
svm = SVC(kernel=kern)
svm.fit(X_red, Y)
Y_test = svm.predict(X_red_test)

cnt, tot = 0, len(Y_test)
for i in range(len(Y_test)):
    if Y_test[i] == Y_real_test[i]:
        cnt += 1

print('svm: pca %d' % (ndim))
print('accuracy: %f' % (cnt/tot))
