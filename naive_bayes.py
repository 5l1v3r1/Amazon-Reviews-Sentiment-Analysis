from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


def run_naive_bayes(X_train, X_test, y_train, y_test):

    # Naive Bayes
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)

    # predict class
    y_pred = classifier.predict(X_test)

    # Confusion matrix
    print("NAIVE BAYES")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print(accuracy_score(y_test, y_pred))
