from sklearn import svm
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from data import y_to_float

from plot import plot_learning_curve, plot_roc_curve, plot_pr_curve


def run_svm(X_train, X_test, y_train, y_test):
    y_train_f = y_to_float(y_train)
    y_test_f = y_to_float(y_test)

    classifier = svm.LinearSVC(random_state=0, max_iter=1000)
    classifier.fit(X_train, y_train_f)
    y_pred = classifier.predict(X_test)
    y_score = classifier.decision_function(X_test)

    print("SVM")
    print(confusion_matrix(y_test_f, y_pred))
    print(classification_report(y_test_f, y_pred))
    print(accuracy_score(y_test_f, y_pred))

    plot_learning_curve(classifier, "SVM Learning Curve", X_train, y_train_f, ylim=(0.6, 1.01), cv=5, n_jobs=-1)
    plot_roc_curve("SVM ROC Curve", y_test_f, classifier.decision_function(X_test))
    plot_pr_curve("SVM Precision Recall Curve", y_test_f, y_pred, classifier.decision_function(X_test))