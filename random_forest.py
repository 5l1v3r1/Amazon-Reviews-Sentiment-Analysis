from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from data import y_to_float
from plot import plot_learning_curve, plot_roc_curve, plot_pr_curve


def run_random_forest(X_train, X_test, y_train, y_test):
    y_train_f = y_to_float(y_train)
    y_test_f = y_to_float(y_test)

    classifier = RandomForestClassifier(n_estimators=10, random_state=0, n_jobs=-1)
    classifier.fit(X_train, y_train_f)
    y_pred = classifier.predict(X_test)

    print("Random Forest")
    print(confusion_matrix(y_test_f, y_pred))
    print(classification_report(y_test_f, y_pred))
    print(accuracy_score(y_test_f, y_pred))

    '''
    plot_learning_curve(classifier, "Random Forest Learning Curve", X_train, y_train_f, ylim=(0.6, 1.01), cv=5, n_jobs=-1)
    plot_roc_curve("Random Forest ROC Curve", y_test_f, classifier.predict_proba(X_test)[:, 1])
    plot_pr_curve("Random Forest Precision Recall Curve", y_test_f, y_pred, classifier.predict_proba(X_test)[:, 1])
    '''