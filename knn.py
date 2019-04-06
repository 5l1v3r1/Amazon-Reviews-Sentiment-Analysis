from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier


def run_knn(data):

    # https://www.youtube.com/watch?v=s-9Qqpv2hTY
    # creating the feature matrix
    matrix = CountVectorizer(max_features=1000)
    X = matrix.fit_transform(data.iloc[:, -1].astype('U')).toarray()
    y_tmp = data.iloc[:, 4]
    y = []
    for idx, val in y_tmp.iteritems():
        if val == 4.0 or val == 5.0:
            y.append("positive")
        elif val == 1.0 or val == 2.0 or val == 3.0:
            y.append("negative")

    # split train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    clf = KNeighborsClassifier(n_neighbors=3, n_jobs=-1)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print("k-NN")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print(accuracy_score(y_test, y_pred))
