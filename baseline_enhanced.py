#!/usr/bin/env python

"""Uses multiple sklearn ML algorithms to classify review categories based on their textual contents."""

import time
import spacy
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV

from evaluation import evaluate_model
from preprocessing import clean_dataframe

spacy.load("en_core_web_sm")


def naive_bayes(features, x_train, y_train, x_test, x_dev, y_dev):
    """Run Naive Bayes classification on an input vector."""
    classifier = Pipeline([('features', features), ('cls', MultinomialNB(alpha=1, fit_prior=True))])
    classifier.fit(x_train, y_train)
    #xxx = tune_parameters(classifier, x_dev, y_dev)
    y_pred = classifier.predict(x_test)
    return y_pred


def tune_parameters(classifier, x_dev, y_dev):
    """" Find the best parameters using GridSearchCV"""
    # Create a dictionary of possible parameters
    print("Tuning parameters. This might take a while...")
    params_grid = {'cls__alpha': [0.00001, 0.0001, 0.001, 0.1, 1, 10, 100, 1000],'cls__fit_prior': [True, False]}

    # Create the GridSearchCV object
    grid_clf = GridSearchCV(estimator=classifier, param_grid=params_grid, scoring='accuracy')

    # Fit the data with the best possible parameters
    grid_clf.fit(x_dev, y_dev)

    # Print the best estimator with it's parameters
    return grid_clf.best_params_


def get_POS_tags(txt):
    """Function to return the POS tags of a sentence, using Spacy"""
    return [token.pos_ for token in nlp(txt)]


def construct_features():
    """Constructs a FeatureUnion of multiple features."""
    # Unigram/Bigram/Trigram
    uni_bi_trigram = CountVectorizer(analyzer='word', ngram_range=(1, 3), token_pattern=r'\b\w+\b', max_features=5000)
    # TFIDF
    tfidf = TfidfVectorizer(analyzer='word', token_pattern=r'\b\w+\b', max_features=5000)
    # Part-Of-Speech tags
    pos = CountVectorizer(tokenizer=get_POS_tags, analyzer='word', token_pattern=r'\b\w+\b', max_features=5000)
    # 3-6 character N-grams
    ngram = CountVectorizer(analyzer='char_wb', ngram_range=(3, 6), min_df=3)

    # Construct a union of features
    union = FeatureUnion([('uni_bi_trigram', uni_bi_trigram), ('tfidf', tfidf), ('pos', pos), ('ngram', ngram)])
    return union


if __name__ == "__main__":
    # Load SpaCy language model
    nlp = spacy.load('en_core_web_sm')

    # Load the files
    df_train = pd.read_csv('data/train.tsv', sep='\t', names=['text', 'label'])
    df_dev = pd.read_csv('data/dev.tsv', sep='\t', names=['text', 'label'])
    df_test = pd.read_csv('data/test.tsv', sep='\t', names=['text', 'label'])

    # Preprocessing the data
    df_train['text'] = clean_dataframe(df_train['text'])
    df_test['text'] = clean_dataframe(df_test['text'])
    df_dev['text'] = clean_dataframe(df_dev['text'])

    # Construct a set of features
    features = construct_features()

    naive_bayes_pred = naive_bayes(features, df_train['text'], df_train['label'], df_test['text'],
                                                   df_dev['text'], df_dev['label'])

    print(evaluate_model(df_test['label'], naive_bayes_pred))