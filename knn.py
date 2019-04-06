from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn.neighbors import KNeighborsClassifier


def run_knn(X_train, X_test, y_train, y_test):

    # https://www.youtube.com/watch?v=s-9Qqpv2hTY

    clf = KNeighborsClassifier(n_neighbors=3, n_jobs=-1)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print("k-NN")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print(accuracy_score(y_test, y_pred))
