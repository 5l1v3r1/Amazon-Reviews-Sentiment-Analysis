from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.neural_network import MLPClassifier

from data import y_to_float


def run_neural_network(X_train, X_test, y_train, y_test):
    y_train_f = y_to_float(y_train)
    y_test_f = y_to_float(y_test)

    classifier = MLPClassifier(alpha=1, max_iter=1000)
    classifier.fit(X_train, y_train_f)

    y_pred = classifier.predict(X_test)

    print("Neural Network")
    print(confusion_matrix(y_test_f, y_pred))
    print(classification_report(y_test_f, y_pred))
    print(accuracy_score(y_test_f, y_pred))
