from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, mean_squared_error, r2_score
from data import y_to_float
import matplotlib.pyplot as plt

#https://towardsdatascience.com/why-linear-regression-is-not-suitable-for-binary-classification-c64457be8e28

def run_linear_regression(X_train, X_test, y_train, y_test):
    y_train_f = y_to_float(y_train)
    y_test_f = y_to_float(y_test)

    regr = LinearRegression(n_jobs=-1)
    regr.fit(X_train, y_train_f)

    y_pred = regr.predict(X_test)
    y_pred = y_pred.round().astype(int)

    print("LINEAR REGRESSION")
    print(confusion_matrix(y_test_f, y_pred))
    print(classification_report(y_test_f, y_pred))
    print(accuracy_score(y_test_f, y_pred))

    # The coefficients
    #print('Coefficients: \n', regr.coef_)
    # The mean squared error
    print("Mean squared error: %.2f"
          % mean_squared_error(y_test_f, y_pred))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(y_test_f, y_pred))

    fig, ax = plt.subplots()
    ax.scatter(y_test_f, y_pred)
    ax.plot([min(y_test_f), max(y_test_f)], [min(y_pred), max(y_pred)], 'k--', lw=4)
    ax.set_xlabel('measured')
    ax.set_ylabel('predicted')
    plt.show()
