from load import *
from sklearn import decomposition
import sys

import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from keras.utils import np_utils

# pca feature dimension
ndim = 100
if len(sys.argv) != 1:
    ndim = int(sys.argv[1])
# label dimension
ldim = 43
batch_size = 64
nb_epoch = 30
dropout_rate = 0.5
hidden_width = 512

print "Load begin"
X_train = load("../data/train.txt").astype('float32')
X_vali = load("../data/vali.txt").astype('float32')
X_test = load("../data/test.txt").astype('float32')
Y_train = load("../data/train_label.txt").flatten().astype('float32')
Y_vali = load("../data/vali_label.txt").flatten().astype('float32')
Y_test = load("../data/test_label.txt").flatten().astype('float32')

Y_train = np_utils.to_categorical(Y_train, ldim)
Y_vali = np_utils.to_categorical(Y_vali, ldim)
Y_test = np_utils.to_categorical(Y_test, ldim)


# pca to reduce dimention
print "PCA begin"
pca = decomposition.PCA(n_components=ndim)
pca.fit(X_train)
X_train = pca.transform(X_train)
X_vali = pca.transform(X_vali)
X_test = pca.transform(X_test)

# normalization to mean=0, var=1
X_train = normalize(X_train)
X_vali = normalize(X_vali)
X_test = normalize(X_test)

# writ("train.txt", X_train)
# writ("vali.txt", X_vali)
# writ("test.txt", X_test)
# 
# X_train = load("./train.txt").astype('float32')
# X_vali = load("./vali.txt").astype('float32')
# X_test = load("./test.txt").astype('float32')

print "Network building begin"
model = Sequential()
model.add(Dense(hidden_width, input_shape=(ndim,), activation='sigmoid'))
model.add(Dropout(dropout_rate))
model.add(Dense(hidden_width, activation='sigmoid'))
model.add(Dropout(dropout_rate))
model.add(Dense(hidden_width, activation='sigmoid'))
model.add(Dropout(dropout_rate))
model.add(Dense(ldim, activation='softmax'))

# print model structure
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

print "Network training begin"
history = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=2, validation_data=(X_vali, Y_vali))

score = model.evaluate(X_test, Y_test, verbose=1)
print "\nTest accuracy: ", score[1]
