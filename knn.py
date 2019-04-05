import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

dataset = pd.read_csv('dataset/foods_clean.csv', sep=";", quotechar="|", encoding='ISO-8859-1',
                      names=['idx', 'productid', 'userid', 'helpfulness', 'rating', 'time', 'summary', 'text', 'cleantext'],
                      header=None)


def balance():
    print(dataset[dataset['rating'] == 5.0].shape[0])
    print(dataset[dataset['rating'] == 4.0].shape[0])
    class_positive = dataset[dataset['rating'] == 5.0]
    class_positive = class_positive.append(dataset[dataset['rating'] == 4.0], ignore_index=True)
    print(class_positive.shape[0])

    class_negative = dataset[dataset['rating'] == 1.0]
    class_negative = class_negative.append(dataset[dataset['rating'] == 2.0], ignore_index=True)
    class_negative = class_negative.append(dataset[dataset['rating'] == 3.0], ignore_index=True)

    positive_under = class_positive.sample(class_negative.shape[0])
    print(positive_under.shape[0])
    print(class_negative.shape[0])
    df_test_under = pd.concat([positive_under, class_negative], axis=0)
    print(df_test_under.shape[0])
    return df_test_under


def run_knn():
    # https://www.youtube.com/watch?v=s-9Qqpv2hTY

    data = balance()
    
    # creating the feature matrix
    matrix = CountVectorizer(max_features=1000)
    X = matrix.fit_transform(data.values.astype('U')).toarray()
    dataset.ix[:, 4] = dataset.ix[:, 4].apply(pd.to_numeric)
    y = dataset.iloc[:, 4]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =.33)
    positive_under = class_positive.sample(class_negative.shape[0])
    print(positive_under.shape[0])
    print(class_negative.shape[0])
    df_test_under = pd.concat([positive_under, class_negative], axis=0)
    print(df_test_under.shape[0])
    return df_test_under

    clf = KNeighborsClassifier()
    clf.fit(X_train, y_train)

    y_expect = y_test
    y_pred = clf.predict(X_test)

    print ( metrics.classification_report(y_expect, y_pred))


if __name__ == '__main__':
    p = 0
    n = 0
    for idx, val in dataset['rating'].iteritems():
        if val == 5.0 or val == 4.0:
            p += 1
        else:
            n += 1
    print(p)
    print(n)
    run_knn()