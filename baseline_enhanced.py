import time
import spacy
import pandas as pd
import argparse

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
    y_pred = classifier.predict(x_test)
    return y_pred


def tune_parameters(classifier, x_dev, y_dev):
    """" Find the best parameters using GridSearchCV"""
    # Create a dictionary of possible parameters
    print("Tuning parameters. This might take a while...")
    params_grid = {'cls__alpha': [0.00001, 0.0001, 0.001, 0.1, 1],'cls__fit_prior': [True, False]}

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
    # Part-Of-Speech tags
    pos = CountVectorizer(tokenizer=get_POS_tags, analyzer='word', token_pattern=r'\b\w+\b', max_features=5000)
    # 3-6 character N-grams
    ngram = CountVectorizer(analyzer='char_wb', ngram_range=(3, 6), min_df=3)

    # Construct a union of features
    union = FeatureUnion([('uni_bi_trigram', uni_bi_trigram), ('pos', pos), ('ngram', ngram)])
    return union


if __name__ == "__main__":
    # Load SpaCy language model
    nlp = spacy.load('en_core_web_sm')
    
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
    
    df_train = pd.read_csv(args.train_file, sep=',', names=['text', 'label'])
    df_dev = pd.read_csv(args.dev_file, sep=',', names=['text', 'label'])
    df_test = pd.read_csv(args.test_file, sep=',', names=['text', 'label'])
    
    # Deleting empty data
    df_train.dropna()
    df_dev.dropna()
    df_test.dropna()

    df_train['text'] = df_train['text'].astype(str)
    df_dev['text'] = df_dev['text'].astype(str)
    df_test['text'] = df_test['text'].astype(str)
    
    

    # Construct a set of features
    features = construct_features()

    naive_bayes_pred = naive_bayes(features, df_train['text'], df_train['label'], df_test['text'],
                                                   df_dev['text'], df_dev['label'])
    
    # Give the evaluation of the model
    print(evaluate_model(df_test['label'], naive_bayes_pred))
