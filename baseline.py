import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

from evaluation import evaluate_model
from preprocessing import clean_dataframe
from preprocessing import data_vectorizer

np.random.seed(1234)


def logistic_regression(x_train, y_train, x_test, y_test, le):
    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    y_pred = le.inverse_transform(y_pred)

    return evaluate_model(y_test, y_pred)


def naive_bayes(x_train, y_train, x_test, y_test, le):
    model = MultinomialNB()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    y_pred = le.inverse_transform(y_pred)

    return evaluate_model(y_test, y_pred)


def linear_svm(x_train, y_train, x_test, y_test, le):
    model = LinearSVC()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    y_pred = le.inverse_transform(y_pred)

    return evaluate_model(y_test, y_pred)


def svmclassifier(x_train, y_train, x_test, y_test, le):
    model = svm.SVC()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    y_pred = le.inverse_transform(y_pred)

    return evaluate_model(y_test, y_pred)


def knnclassifier(x_train, y_train, x_test, y_test, le):
    model = KNeighborsClassifier()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    y_pred = le.inverse_transform(y_pred)

    return evaluate_model(y_test, y_pred)


def randomforestclassifier(x_train, y_train, x_test, y_test, le):
    model = RandomForestClassifier()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    y_pred = le.inverse_transform(y_pred)

    return evaluate_model(y_test, y_pred)


def decisiontreeclassifier(x_train, y_train, x_test, y_test, le):
    model = DecisionTreeClassifier()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    y_pred = le.inverse_transform(y_pred)

    return evaluate_model(y_test, y_pred)


def print_all_baseline_results(x_train, y_train, x_test, y_test, le):
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
    df_train = pd.read_csv('preprocessed_data/train.csv', names=['text', 'label'])
    df_dev = pd.read_csv('preprocessed_data/dev.csv', names=['text', 'label'])
    df_test = pd.read_csv('preprocessed_data/test.csv', names=['text', 'label'])

    df_train['text'] = df_train['text'].astype(str)
    df_dev['text'] = df_dev['text'].astype(str)
    df_test['text'] = df_test['text'].astype(str)

    # Preprocessing the data
    df_train['text'] = clean_dataframe(df_train['text'])
    df_test['text'] = clean_dataframe(df_test['text'])
    df_dev['text'] = clean_dataframe(df_dev['text'])

    x_train, y_train, x_test, y_test, le = data_vectorizer(df_train['text'], df_train['label'], df_test['text'],
                                                           df_test['label'])

    print_all_baseline_results(x_train, y_train, x_test, y_test, le)
