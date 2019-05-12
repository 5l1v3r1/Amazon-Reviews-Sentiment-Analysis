from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import learning_curve
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_curve, \
    average_precision_score
import matplotlib.pyplot as plt
from sklearn.utils.fixes import signature
import numpy as np

from data import y_to_float
from plot import plot_learning_curve, plot_roc_curve, plot_pr_curve

'''
def plot(y_test_f, y_pred):
    precision, recall, _ = precision_recall_curve(y_test_f, y_pred)
    print("Precision: ", precision)
    print("Recall: ", recall)
    # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
    step_kwargs = ({'step': 'post'}
                   if 'step' in signature(plt.fill_between).parameters
                   else {})
    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

    average_precision = average_precision_score(y_test_f, y_pred)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
        average_precision))
    plt.show()
'''


def run_gaussianNB(X_train, X_test, y_train, y_test):
    y_train_f = y_to_float(y_train)
    y_test_f = y_to_float(y_test)

    # Naive Bayes
    classifier = GaussianNB()
    classifier.fit(X_train, y_train_f)

    # predict class
    y_pred = classifier.predict(X_test)

    # Confusion matrix
    print("GAUSSIAN NAIVE BAYES")
    print(confusion_matrix(y_test_f, y_pred))
    print(classification_report(y_test_f, y_pred))
    print(accuracy_score(y_test_f, y_pred))

    # plot_learning_curve(classifier, "Gaussian Naive Bayes Learning Curve", X_train, y_train_f, ylim=(0.6, 1.01), cv=5, n_jobs=-1)
    # plot_roc_curve("Gaussian Naive Bayes ROC Curve", y_test_f, classifier.predict_proba(X_test)[:, 1])
    plot_pr_curve("Gaussian Naive Bayes Precision Recall Curve", y_test_f, y_pred,
                  classifier.predict_proba(X_test)[:, 1])


def run_multinomialNB(X_train, X_test, y_train, y_test):
    y_train_f = y_to_float(y_train)
    y_test_f = y_to_float(y_test)
    tf_transformer = TfidfTransformer().fit_transform(X_train)

    classifier = MultinomialNB()
    classifier.fit(tf_transformer, y_train_f)

    y_pred = classifier.predict(X_test)

    # Confusion matrix
    print("MULTINOMIAL NAIVE BAYES")
    print(confusion_matrix(y_test_f, y_pred))
    print(classification_report(y_test_f, y_pred))
    print(accuracy_score(y_test_f, y_pred))

    # plot_learning_curve(classifier, "Multinomial Naive Bayes Learning Curve", X_train, y_train_f, ylim=(0.6, 1.01), cv=5, n_jobs=-1)
    # plot_roc_curve("Multinomial Naive Bayes ROC Curve", y_test_f, classifier.predict_proba(X_test)[:, 1])
    plot_pr_curve("Multinomial Naive Bayes Precision Recall Curve", y_test_f, y_pred, classifier.predict_proba(X_test)[:, 1])


def run_bernoulliNB(X_train, X_test, y_train, y_test):
    y_train_f = y_to_float(y_train)
    y_test_f = y_to_float(y_test)

    classifier = BernoulliNB()
    classifier.fit(X_train, y_train_f)

    y_pred = classifier.predict(X_test)

    # Confusion matrix
    print("BERNOULLI NAIVE BAYES")
    print(confusion_matrix(y_test_f, y_pred))
    print(classification_report(y_test_f, y_pred))
    print(accuracy_score(y_test_f, y_pred))

    # plot_learning_curve(classifier, "Bernoulli Naive Bayes Learning Curve", X_train, y_train_f, ylim=(0.6, 1.01), cv=5, n_jobs=-1)
    # plot_roc_curve("Bernoulli Naive Bayes ROC Curve", y_test_f, classifier.predict_proba(X_test)[:, 1])
    plot_pr_curve("Bernoulli Naive Bayes Precision Recall Curve", y_test_f, y_pred, classifier.predict_proba(X_test)[:, 1])

