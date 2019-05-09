from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')


def run_knn(X_train, X_test, y_train, y_test):
    neighbors = np.arange(2, 10)
    train_accuracy = np.empty(len(neighbors))
    test_accuracy = np.empty(len(neighbors))

    for i, k in enumerate(neighbors):
        print(i)
        print(k)
        # Setup a knn classifier with k neighbors
        knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)

        # Fit the model
        knn.fit(X_train, y_train)

        # Compute accuracy on the training set
        train_accuracy[i] = knn.score(X_train, y_train)
        print("Train Acc: ", train_accuracy[i])

        # Compute accuracy on the test set
        test_accuracy[i] = knn.score(X_test, y_test)
        print("Test Acc: ", train_accuracy[i])

    # Generate plot
    plt.title('k-NN Varying number of neighbors')
    plt.plot(neighbors, test_accuracy, label='Testing Accuracy')
    plt.plot(neighbors, train_accuracy, label='Training accuracy')
    plt.legend()
    plt.xlabel('Number of neighbors')
    plt.ylabel('Accuracy')
    plt.show()

    # https://www.youtube.com/watch?v=s-9Qqpv2hTY


'''
    clf = KNeighborsClassifier(n_neighbors=2, n_jobs=-1)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print("k-NN")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print(accuracy_score(y_test, y_pred))
'''
