from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from data import y_to_float
from naive_bayes import plot


def run_logistic_regression(X_train, X_test, y_train, y_test):
    y_train_f = y_to_float(y_train)
    y_test_f = y_to_float(y_test)

    # see https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    # to use different solvers and classes
    regr = LogisticRegression(solver='saga', multi_class='ovr', n_jobs=-1)
    regr.fit(X_train, y_train_f)

    y_pred = regr.predict(X_test)
    y_pred = y_pred.round().astype(int)

    y_score = regr.decision_function(X_test)

    print("LOGISTIC REGRESSION")
    print(confusion_matrix(y_test_f, y_pred))
    print(classification_report(y_test_f, y_pred))
    print(accuracy_score(y_test_f, y_pred))

    plot(y_test_f, y_score)