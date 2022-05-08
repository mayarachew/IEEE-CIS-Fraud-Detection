"""Function to test hyperparameters to improve classification models."""

from sklearn.naive_bayes import MultinomialNB # type: ignore
from sklearn.ensemble import RandomForestClassifier # type: ignore
from sklearn.svm import SVC # type: ignore
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV # type: ignore

SEED_VAL = 42

def test_hyperparameters(classifier, X_train, y_train):
    parameters = {}
    classification = None

    if classifier == 'Random Forest':
        parameters = {'n_estimators': [200, 400, 600, 800, 1000], 
                     'max_depth': [10, 20, 40, 60, 80, 100, None], 
                     'criterion': ['gini', 'entropy'], 
                     'max_features': ['auto', 'sqrt', 'log2'],
                     'min_samples_leaf': [1, 2, 4],
                     'min_samples_split': [2, 5, 10]}
        classification = RandomForestClassifier(random_state=SEED_VAL)

    elif classifier == 'Naive Bayes':
        parameters = {'alpha': [1, 0.1, 0.01, 0.001, 0.0001]}
        classification = MultinomialNB()

    elif classifier == 'SVM':
        parameters = {'kernel': ['rbf', 'poly', 'sigmoid'], 
                      "C": [1, 10, 100, 1000]}
        classification = SVC(random_state=SEED_VAL)
        
    elif classifier == 'Multilayer Perceptron':
        parameters = {'hidden_layer_sizes': [(10,), (20,), (40,), (80,), (100,), (300,), (500,)],
                      'activation': ['tanh', 'relu', 'sigmoid'],
                      'solver': ['sgd', 'adam'],
                      'learning_rate': ['constant', 'adaptative'],
                      'alpha': [1, 0.1, 0.01, 0.001, 0.0001]}
        classification = MLPClassifier(random_state=SEED_VAL)

    score = 'f1'

    clf = RandomizedSearchCV(classification, parameters, scoring="%s_macro" % score, cv=5)
    clf.fit(X_train, y_train)

    print(f"Best hyperparameters based on {score} score for {classifier}: {clf.best_params_}")

def test_hyperparameters_grid(classifier, X_train, y_train):
    parameters = {}
    classification = None

    if classifier == 'Random Forest':
        parameters = {'n_estimators': [200, 400, 600, 800, 1000], 
                     'max_depth': [10, 20, 40, 60, 80, 100, None], 
                     'criterion': ['gini', 'entropy'], 
                     'max_features': ['auto', 'sqrt', 'log2'],
                     'min_samples_leaf': [1, 2, 4],
                     'min_samples_split': [2, 5, 10]}
        classification = RandomForestClassifier(random_state=SEED_VAL)

    elif classifier == 'Naive Bayes':
        parameters = {'alpha': [1, 0.1, 0.01, 0.001, 0.0001]}
        classification = MultinomialNB()

    elif classifier == 'SVM':
        parameters = {'kernel': ['rbf', 'poly', 'sigmoid'], 
                      "C": [1, 10, 100, 1000]}
        classification = SVC(random_state=SEED_VAL)
        
    elif classifier == 'Multilayer Perceptron':
        parameters = {'hidden_layer_sizes': [(10,), (20,), (40,), (80,), (100,), (300,), (500,)],
                      'activation': ['tanh', 'relu', 'sigmoid'],
                      'solver': ['sgd', 'adam'],
                      'learning_rate': ['constant', 'adaptative'],
                      'alpha': [1, 0.1, 0.01, 0.001, 0.0001]}
        classification = MLPClassifier(random_state=SEED_VAL)

    score = 'f1'

    clf = GridSearchCV(classification, parameters, scoring="%s_macro" % score, cv=5)
    clf.fit(X_train, y_train)

    print(f"Best hyperparameters based on {score} score for {classifier}: {clf.best_params_}")