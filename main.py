from data import get_data_frame, split_train_test
from knn import run_knn
from naive_bayes import run_naive_bayes

if __name__ == "__main__":
    data = get_data_frame()
    X_train, X_test, y_train, y_test = split_train_test(data)

    run_naive_bayes(X_train, X_test, y_train, y_test)
    # run_knn(data)
