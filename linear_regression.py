from sklearn.linear_model import LinearRegression
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


def run_linear_regression(X_train, X_test, y_train, y_test):
    y_train_f = y_to_float(y_train)
    y_test_f = y_to_float(y_test)

    regr = LinearRegression()
    regr.fit(X_train, y_train_f)

    y_pred = regr.predict(X_test)

    print("LINEAR REGRESSION")
    print(confusion_matrix(y_test_f, y_pred.round()))
    print(classification_report(y_test_f, y_pred.round()))
    print(accuracy_score(y_test_f, y_pred.round()))
