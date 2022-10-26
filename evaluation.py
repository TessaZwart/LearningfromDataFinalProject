from sklearn.metrics import classification_report


def evaluate_model(y_test, y_pred):

    return classification_report(list(y_test), list(y_pred), labels=['NOT', 'OFF'],digits=4)

