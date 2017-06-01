from load import load
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
batch_size = 128
nb_epoch = 10

X_train = load("../data/train.txt").astype('float32')
Y_train = load("../data/label.txt").flatten().astype('float32')
X_vali = load("../data/vali.txt").astype('float32')
Y_vali = load("../data/vali_label.txt").flatten().astype('float32')
X_test = load("../data/test.txt").astype('float32')
Y_test = load("../data/test_label.txt").flatten().astype('float32')

# pca to reduce dimention
pca = decomposition.PCA(n_components=ndim)
pca.fit(X_train)
X_train = pca.transform(X_train)
X_vali = pca.transform(X_vali)
X_test = pca.transform(X_test)

model = Sequential()
model.add(Dense(512, input_shape=(ndim,), activation='sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(512, activation='sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(ldim, activation='softmax'))

# print model structure
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_data=(X_vali, Y_vali))

score = model.evaluate(X_test, Y_test, verbose=1)
print "Test accuracy: ", score[1]
