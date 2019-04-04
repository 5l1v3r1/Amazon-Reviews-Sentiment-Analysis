import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

dataset = pd.read_csv('dataset/foods_clean.csv', sep=";", quotechar="|", encoding='ISO-8859-1')

data = dataset.iloc[:, -1]


def run_naive_bayes():
    # creating the feature matrix
    matrix = CountVectorizer(max_features=1000)
    X = matrix.fit_transform(data.values.astype('U')).toarray()
    dataset.ix[:, 4] = dataset.ix[:, 4].apply(pd.to_numeric)
    y = dataset.iloc[:, 4]

    # split train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # Naive Bayes
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)

    # predict class
    y_pred = classifier.predict(X_test)

    # Confusion matrix

    cm = confusion_matrix(y_test, y_pred)
    print cm
    cr = classification_report(y_test, y_pred)
    print cr

    accuracy = accuracy_score(y_test, y_pred)
    print accuracy


if __name__ == '__main__':
    run_naive_bayes()
