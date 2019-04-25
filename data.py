import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split


def get_data_frame():
    dataset = pd.read_csv('dataset/foods.csv', sep=";", quotechar="|", encoding='ISO-8859-1',
                          names=['productid', 'userid', 'helpfulness', 'rating', 'time', 'summary',
                                 'text'],
                          header=None)

    # get positive class
    class_positive = dataset[dataset['rating'] == 5.0]
    class_positive = class_positive.append(dataset[dataset['rating'] == 4.0], ignore_index=True)

    # get negative class
    class_negative = dataset[dataset['rating'] == 1.0]
    class_negative = class_negative.append(dataset[dataset['rating'] == 2.0], ignore_index=True)
    class_negative = class_negative.append(dataset[dataset['rating'] == 3.0], ignore_index=True)

    # under sample positive class randomly with the size of negative class
    positive_under = class_positive.sample(class_negative.shape[0])
    # concatenate negative and new under sampled positive class
    df_test_under = pd.concat([positive_under, class_negative], axis=0)

    return df_test_under


def split_train_test(data):
    # creating the feature matrix
    matrix = CountVectorizer(max_features=1000, stop_words="english")
    X = matrix.fit_transform(data.iloc[:, -1].astype('U')).toarray()
    y_tmp = data.iloc[:, 3]

    y = []
    for idx, val in y_tmp.iteritems():
        if val == 4.0 or val == 5.0:
            y.append("positive")
        elif val == 1.0 or val == 2.0 or val == 3.0:
            y.append("negative")

    # split train and test data
    return train_test_split(X, y)


# linear/logistic regression and random forest accepts numeric
# so prediction vector y changed as 0/1
def y_to_float(y):
    y_float = []
    for val in y:
        if val == "positive":
            y_float.append(1)
        elif val == "negative":
            y_float.append(0)

    return y_float
