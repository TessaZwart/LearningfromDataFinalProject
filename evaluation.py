from sklearn.metrics import classification_report


def evaluate_model(y_test, y_pred):
    return classification_report(y_test, y_pred, labels=['Offensive', 'Non-offensive'])

