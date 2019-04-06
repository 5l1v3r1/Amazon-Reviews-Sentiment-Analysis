from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


def run_gaussianNB(X_train, X_test, y_train, y_test):
    # Naive Bayes
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)

    # predict class
    y_pred = classifier.predict(X_test)

    # Confusion matrix
    print("GAUSSIAN NAIVE BAYES")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print(accuracy_score(y_test, y_pred))


def run_multinomialNB(X_train, X_test, y_train, y_test):
    tf_transformer = TfidfTransformer().fit_transform(X_train)

    classifier = MultinomialNB()
    classifier.fit(tf_transformer, y_train)

    y_pred = classifier.predict(X_test)

    # Confusion matrix
    print("MULTINOMIAL NAIVE BAYES")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print(accuracy_score(y_test, y_pred))


def run_bernoulliNB(X_train, X_test, y_train, y_test):
    classifier = BernoulliNB()
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    # Confusion matrix
    print("BERNOULLI NAIVE BAYES")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print(accuracy_score(y_test, y_pred))
