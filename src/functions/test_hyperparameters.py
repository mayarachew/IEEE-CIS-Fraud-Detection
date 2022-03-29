"""Function to test hyperparameters to improve classification models."""

from sklearn.naive_bayes import MultinomialNB # type: ignore
from sklearn.ensemble import RandomForestClassifier # type: ignore
from sklearn.svm import SVC # type: ignore
from sklearn.model_selection import RandomizedSearchCV # type: ignore


def test_hyperparameters(classifier, X_train, y_train):
    parameters = {}
    classification = None

    if classifier == 'Random Forest':
        parameters = {'n_estimators': [100, 300, 500], "max_depth": [
            3, 5, 7, None], 'criterion': ['gini', 'entropy'], 'max_features': ['auto', 'sqrt', 'log2']}
        classification = RandomForestClassifier(random_state=0)
    elif classifier == 'Naive Bayes':
        parameters = {'alpha': [1, 0.1, 0.01, 0.001, 0.0001]}
        classification = MultinomialNB()
    elif classifier == 'Support Vector Machine':
        parameters = {'kernel': ['rbf', 'poly',
                                 'sigmoid', 'linear'], "C": [1, 10, 100, 1000]}
        classification = SVC(random_state=0)

    score = 'f1'

    print("# Defining hyperparameters based on %s" % score)
    print()

    clf = RandomizedSearchCV(classification, parameters,
                             scoring="%s_macro" % score, cv=5, random_state=0)
    clf.fit(X_train, y_train)

    print("Best hyperparameters:")
    print()
    print(clf.best_params_)