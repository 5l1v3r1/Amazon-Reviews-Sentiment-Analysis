from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
from data import y_to_float
from plot import plot_pr_curve


def run_logistic_regression(X_train, X_test, y_train, y_test):
    y_train_f = y_to_float(y_train)
    y_test_f = y_to_float(y_test)

    # see https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    # to use different solvers and classes
    regr = LogisticRegression(solver='liblinear', multi_class='ovr', n_jobs=-1, max_iter=200)
    regr.fit(X_train, y_train_f)

    y_pred = regr.predict(X_test)
    y_pred = y_pred.round().astype(int)

    print("LOGISTIC REGRESSION")
    print(confusion_matrix(y_test_f, y_pred))
    print(classification_report(y_test_f, y_pred))
    print(accuracy_score(y_test_f, y_pred))

    '''
    plt.title("Logistic Regression")
    plt.scatter(regr.decision_function(X_test), regr.predict_proba(X_test)[:,1])
    plt.xlabel("Features")
    plt.ylabel("Probability")
    plt.show()
    '''