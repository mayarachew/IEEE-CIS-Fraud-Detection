"""Functions to create the classifier and plot results"""

from sklearn.metrics import classification_report # type: ignore
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt # type: ignore


def create_classifier(classifier, x_train, y_train, x_test, y_test):
    # Create classifier
    classifier.fit(x_train, y_train)

    # Define test labels
    y_true, y_pred = y_test, classifier.predict(x_test)

    print('Classification report: ')
    print(classification_report(y_true, y_pred, zero_division=1))

    return y_test


def plot_confusion_matrix(classifier, x_test, y_test):
    ConfusionMatrixDisplay.from_estimator(
        classifier, x_test, y_test, cmap=plt.cm.Blues)
    plt.show()

    return