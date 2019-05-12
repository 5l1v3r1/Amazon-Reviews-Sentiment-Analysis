import multiprocessing

from ada_boost import run_ada_boost
from data import get_data_frame, split_train_test
from decision_tree import run_decision_tree
from knn import run_knn
from linear_regression import run_linear_regression
from logistic_regression import run_logistic_regression
from naive_bayes import run_gaussianNB, run_multinomialNB, run_bernoulliNB
from neural_network import run_neural_network
from random_forest import run_random_forest
from svm import run_svm

if __name__ == "__main__":
    data = get_data_frame()
    X_train, X_test, y_train, y_test = split_train_test(data)

    # Uncomment the algorithms to run as you wish
    # be careful with knn it is too slow.

    run_gaussianNB(X_train, X_test, y_train, y_test)
    run_multinomialNB(X_train, X_test, y_train, y_test)
    run_bernoulliNB(X_train, X_test, y_train, y_test)
    #run_knn(X_train, X_test, y_train, y_test)
    #run_linear_regression(X_train, X_test, y_train, y_test)
    #run_logistic_regression(X_train, X_test, y_train, y_test)
    #run_svm(X_train, X_test, y_train, y_test)
    #run_random_forest(X_train, X_test, y_train, y_test)
    #run_decision_tree(X_train, X_test, y_train, y_test)
    #run_neural_network(X_train, X_test, y_train, y_test)
    #run_ada_boost(X_train, X_test, y_train, y_test)
