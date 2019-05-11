from sklearn import svm
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from data import y_to_float
from naive_bayes import plot


def run_svm(X_train, X_test, y_train, y_test):
    y_train_f = y_to_float(y_train)
    y_test_f = y_to_float(y_test)

    classifier = svm.LinearSVC(random_state=0)
    classifier.fit(X_train, y_train_f)
    y_pred = classifier.predict(X_test)
    y_score = classifier.decision_function(X_test)

    print("SVM")
    print(confusion_matrix(y_test_f, y_pred))
    print(classification_report(y_test_f, y_pred))
    print(accuracy_score(y_test_f, y_pred))