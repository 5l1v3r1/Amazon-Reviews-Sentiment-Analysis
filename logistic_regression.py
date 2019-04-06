from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


# linear regression accepts numeric
# so prediction vector y changed as 0/1
def y_to_float(y):
    y_float = []
    for val in y:
        if val == "positive":
            y_float.append(1)
        elif val == "negative":
            y_float.append(0)

    return y_float


def run_logistic_regression(X_train, X_test, y_train, y_test):
    y_train_f = y_to_float(y_train)
    y_test_f = y_to_float(y_test)

    # see https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    # to use different solvers and classes
    regr = LogisticRegression(solver='saga', multi_class='multinomial', n_jobs=-1)
    regr.fit(X_train, y_train_f)

    y_pred = regr.predict(X_test)

    print("LOGISTIC REGRESSION")
    print(confusion_matrix(y_test_f, y_pred.round()))
    print(classification_report(y_test_f, y_pred.round()))
    print(accuracy_score(y_test_f, y_pred.round()))
