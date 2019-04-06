from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from data import y_to_float


def run_linear_regression(X_train, X_test, y_train, y_test):
    y_train_f = y_to_float(y_train)
    y_test_f = y_to_float(y_test)

    regr = LinearRegression(n_jobs=-1)
    regr.fit(X_train, y_train_f)

    y_pred = regr.predict(X_test)

    print("LINEAR REGRESSION")
    print(confusion_matrix(y_test_f, y_pred.round()))
    print(classification_report(y_test_f, y_pred.round()))
    print(accuracy_score(y_test_f, y_pred.round()))
