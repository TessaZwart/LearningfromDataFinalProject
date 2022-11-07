import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import argparse

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

from evaluation import evaluate_model
from preprocessing import data_vectorizer

np.random.seed(1234)



def logistic_regression(x_train, y_train, x_test, y_test, le):
    """ Performs the Logistic regression and print the evaluation """
    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    y_pred = le.inverse_transform(y_pred)

    return evaluate_model(y_test, y_pred)


def naive_bayes(x_train, y_train, x_test, y_test, le):
    """ Performs the Naive Bayes and print the evaluation """
    model = MultinomialNB()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    y_pred = le.inverse_transform(y_pred)

    return evaluate_model(y_test, y_pred)


def linear_svm(x_train, y_train, x_test, y_test, le):
    """ Performs the Linear SVM and print the evaluation """
    model = LinearSVC()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    y_pred = le.inverse_transform(y_pred)

    return evaluate_model(y_test, y_pred)


def svmclassifier(x_train, y_train, x_test, y_test, le):
    """ Performs the SVM and print the evaluation """
    model = svm.SVC()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    y_pred = le.inverse_transform(y_pred)

    return evaluate_model(y_test, y_pred)


def knnclassifier(x_train, y_train, x_test, y_test, le):
    """ Performs the KNN and print the evaluation """
    model = KNeighborsClassifier()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    y_pred = le.inverse_transform(y_pred)

    return evaluate_model(y_test, y_pred)


def randomforestclassifier(x_train, y_train, x_test, y_test, le):
    """ Performs the Random Forest and print the evaluation """
    model = RandomForestClassifier()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    y_pred = le.inverse_transform(y_pred)

    return evaluate_model(y_test, y_pred)


def decisiontreeclassifier(x_train, y_train, x_test, y_test, le):
    """ Performs the Decision Tree and print the evaluation """
    model = DecisionTreeClassifier()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    y_pred = le.inverse_transform(y_pred)

    return evaluate_model(y_test, y_pred)


def print_all_baseline_results(x_train, y_train, x_test, y_test, le):
    """ Prints all the results and there running time """
    start = time.time()
    print("----Logistic Regression ----")
    print(logistic_regression(x_train, y_train, x_test, y_test, le))
    print("Executed in s: " + str(start - time.time()))

    print("----Naive Bayes ----")
    print(naive_bayes(x_train, y_train, x_test, y_test, le))
    print("Executed in s: " + str(start - time.time()))

    print("---- Lineair SVM ----")
    print(linear_svm(x_train, y_train, x_test, y_test, le))
    print("Executed in s: " + str(start - time.time()))

    print("---- SVM ----")
    print(svmclassifier(x_train, y_train, x_test, y_test, le))
    print("Executed in s: " + str(start - time.time()))

    print("---- KNN ----")
    print(knnclassifier(x_train, y_train, x_test, y_test, le))
    print("Executed in s: " + str(start - time.time()))

    print("---- Random Forest ----")
    print(randomforestclassifier(x_train, y_train, x_test, y_test, le))
    print("Executed in s: " + str(start - time.time()))

    print("---- Decision Tree ----")
    print(decisiontreeclassifier(x_train, y_train, x_test, y_test, le))
    print("Executed in s: " + str(start - time.time()))


if __name__ == "__main__":
    # Parser arguments to make it easier to use different data
    parser = argparse.ArgumentParser()
    parser.add_argument("-tf", "--train_file", default='preprocessed_data/train.csv', type=str,
                        help="Train file to learn from (default train.csv)")
    parser.add_argument("-df", "--dev_file", default='preprocessed_data/dev.csv', type=str,
                        help="Dev file to evaluate on (default dev.csv)")
    parser.add_argument("-tsf", "--test_file", default='preprocessed_data/test.csv', type=str,
                        help="Test file to evaluate on (default test.csv)")
    parser.add_argument("-sw", "--stop_words", default=False, type=bool,
                        help="Remove stopwords True/False (default false)")
    args = parser.parse_args()
    
    df_train = pd.read_csv(args.train_file, names=['text', 'label'])
    df_dev = pd.read_csv(args.dev_file, names=['text', 'label'])
    df_test = pd.read_csv(args.test_file, names=['text', 'label'])

    df_train['text'] = df_train['text'].astype(str)
    df_dev['text'] = df_dev['text'].astype(str)
    df_test['text'] = df_test['text'].astype(str)


    x_train, y_train, x_test, y_test, le = data_vectorizer(df_train['text'], df_train['label'], df_test['text'],
                                                           df_test['label'])
    
    # Give the evaluation of the model
    print_all_baseline_results(x_train, y_train, x_test, y_test, le)
