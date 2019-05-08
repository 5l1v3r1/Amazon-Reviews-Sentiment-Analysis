import multiprocessing

from data import get_data_frame, split_train_test
from knn import run_knn
from linear_regression import run_linear_regression
from logistic_regression import run_logistic_regression
from naive_bayes import run_gaussianNB, run_multinomialNB, run_bernoulliNB

if __name__ == "__main__":
    data = get_data_frame()
    X_train, X_test, y_train, y_test = split_train_test(data)

    # Uncomment the algorithms to run as you wish
    # be careful with knn it is too slow.

    #run_gaussianNB(X_train, X_test, y_train, y_test)
    #run_multinomialNB(X_train, X_test, y_train, y_test)
    #run_bernoulliNB(X_train, X_test, y_train, y_test)
    # run_knn(X_train, X_test, y_train, y_test)
    # run_linear_regression(X_train, X_test, y_train, y_test)
    run_logistic_regression(X_train, X_test, y_train, y_test)
