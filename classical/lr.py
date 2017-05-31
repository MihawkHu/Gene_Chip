from sklearn import linear_model


# lr(train_x, train_y, 'l1'/'l2', test_x), output test_y

def logistic_regression(X, Y, norm, test_X):
    logreg = linear_model.LogisticRegression(penalty=norm)
    logreg.fit(X, Y)
    return logreg.predict(test_X)