from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from data import y_to_float


def run_random_forest(X_train, X_test, y_train, y_test):
    y_train_f = y_to_float(y_train)
    y_test_f = y_to_float(y_test)

    regressor = RandomForestClassifier(n_estimators=20, random_state=0, n_jobs=-1)
    regressor.fit(X_train, y_train_f)
    y_pred = regressor.predict(X_test)

    print("Random Forest")
    print(confusion_matrix(y_test_f, y_pred))
    print(classification_report(y_test_f, y_pred))
    print(accuracy_score(y_test_f, y_pred))
