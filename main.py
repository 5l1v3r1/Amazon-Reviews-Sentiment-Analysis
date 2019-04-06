import pandas as pd

from knn import run_knn
from naive_bayes import run_naive_bayes


def get_data_frame():
    dataset = pd.read_csv('dataset/foods_clean.csv', sep=";", quotechar="|", encoding='ISO-8859-1',
                          names=['idx', 'productid', 'userid', 'helpfulness', 'rating', 'time', 'summary',
                                 'text', 'cleantext'],
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


if __name__ == "__main__":
    data = get_data_frame()

    run_naive_bayes(data)
    # run_knn(data)
