import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import nltk

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from evaluation import evaluate_model
from preprocessing import clean_dataframe
from preprocessing import data_vectorizer

df_train = pd.read_csv('data/train.tsv', sep='\t', names=['text', 'label'])
df_dev = pd.read_csv('data/dev.tsv', sep='\t', names=['text', 'label'])
df_test = pd.read_csv('data/test.tsv', sep='\t', names=['text', 'label'])


def logistic_regression(x_train, y_train, x_test, y_test, le):
    model = LogisticRegression(random_state=0, max_iter=1000)
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


def svm(x_train, y_train, x_test, y_test, le):
    model = LinearSVC()
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


def randomforestclassifier(x_train, y_train, x_test, y_test, le):
    model = RandomForestClassifier()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    y_pred = le.inverse_transform(y_pred)

    return evaluate_model(y_test, y_pred)


def print_all_baseline_results(x_train, y_train, x_test, y_test, le):
    print("----Logistic Regression ----")
    print(logistic_regression(x_train, y_train, x_test, y_test, le))
    print("----Naive Bayes ----")
    print(naive_bayes(x_train, y_train, x_test, y_test, le))
    print("---- SVM ----")
    print(svm(x_train, y_train, x_test, y_test, le))
    print("---- Decision Tree ----")
    print(decisiontreeclassifier(x_train, y_train, x_test, y_test, le))
    print("---- Random Forest ----")
    print(randomforestclassifier(x_train, y_train, x_test, y_test, le))


if __name__ == "__main__":
    df_train['text'] = clean_dataframe(df_train['text'])
    df_dev['text'] = clean_dataframe(df_dev['text'])

    x_train, y_train, x_test, y_test, le = data_vectorizer(df_train['text'], df_train['label'], df_test['text'],
                                                           df_test['label'])

    print_all_baseline_results(x_train, y_train, x_test, y_test, le)
