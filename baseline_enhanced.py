#!/usr/bin/env python

"""Uses multiple sklearn ML algorithms to classify review categories based on their textual contents."""

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import spacy


def naive_bayes(features):
    """Run Naive Bayes classification on an input vector."""
    classifier = Pipeline([('features', features), ('cls', MultinomialNB(alpha=0.5, fit_prior=True))])
    classifier.fit(X_train, y_train)
    # tune_parameters(classifier, 'NB')
    y_pred = classifier.predict(X_test)
    return y_pred


def decision_tree(features):
    """Run Decision Tree classification on an input vector."""
    classifier = Pipeline(
        [('features', features), ('cls', DecisionTreeClassifier(criterion='gini', max_depth=10, min_samples_split=3))])
    classifier.fit(X_train, y_train)
    # tune_parameters(classifier, 'DT')
    y_pred = classifier.predict(X_test)
    return y_pred


def random_forest(features):
    """Run Random Forest classification on an input vector."""
    classifier = Pipeline([('features', features), (
    'cls', RandomForestClassifier(criterion='gini', max_depth=20, min_samples_split=2, n_estimators=50))])
    classifier.fit(X_train, y_train)
    # tune_parameters(classifier, 'RF')
    y_pred = classifier.predict(X_test)
    return y_pred


def knn(features):
    """Run KNN classification on an input vector."""
    classifier = Pipeline(
        [('features', features), ('cls', KNeighborsClassifier(algorithm='auto', n_neighbors=8, weights='distance'))])
    classifier.fit(X_train, y_train)
    # tune_parameters(classifier, 'KNN')
    y_pred = classifier.predict(X_test)
    return y_pred


def supvecmac(features):
    """Run SVM classification on an input vector."""
    classifier = Pipeline([('features', features), ('cls', svm.SVC(C=0.1, gamma=0.0001, kernel='linear'))])
    classifier.fit(X_train, y_train)
    # tune_parameters(classifier, 'SVM')
    y_pred = classifier.predict(X_test)
    return y_pred


def linear_svm(features):
    """Run linear SVM classification on an input vector."""
    classifier = Pipeline([('features', features), ('cls', svm.LinearSVC())])
    classifier.fit(X_train, y_train)
    tune_parameters(classifier, 'linear_SVM')

    y_pred = classifier.predict(X_test)
    return y_pred


def tune_parameters(classifier, algorithm):
    """" Find the best parameters using GridSearchCV"""
    # Create a dictionary of possible parameters
    if algorithm == 'NB':
        params_grid = {'cls__alpha': [0.1, 0.5, 1.0],
                       'cls__fit_prior': [True, False]}
    elif algorithm == 'DT':
        params_grid = {'cls__criterion': ['gini', 'entropy'],
                       'cls__max_depth': [10, 20, 40, None],
                       'cls__min_samples_split': [2, 3, 5]}
    elif algorithm == 'RF':
        params_grid = {'cls__criterion': ['gini', 'entropy'],
                       'cls__max_depth': [20, 40, 60, None],
                       'cls__min_samples_split': [2, 3, 5],
                       'cls__n_estimators': [50, 100, 150]}
    elif algorithm == 'KNN':
        params_grid = {'cls__n_neighbors': [2, 5, 8],
                       'cls__weights': ['uniform', 'distance'],
                       'cls__algorithm': ['auto', 'brute']}
    elif algorithm == 'SVM':
        params_grid = {'cls__C': [0.1, 1, 10, 100],
                       'cls__gamma': [0.0001, 0.01, 1, 10],
                       'cls__kernel': ['rbf', 'poly', 'sigmoid', 'linear']}
    elif algorithm == 'linear_SVM':
        params_grid = {'cls__C': [0.1, 1, 10, 100],
                       'cls__max_iter': [500, 1000, 2000],
                       'cls__tol': [0.00001, 0.0001, 0.001]}

    # Create the GridSearchCV object
    grid_clf = GridSearchCV(estimator=classifier, param_grid=params_grid, scoring='accuracy')

    # Fit the data with the best possible parameters
    grid_clf.fit(X_dev, y_dev)

    # Print the best estimator with it's parameters
    print(grid_clf.best_params_)


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


def word_count(txt):
    """Returns the length of a sentence"""
    return len(txt.split())


def create_feature_dict(txt):
    """Create a dictionary of features"""
    dic = {}
    dic["word_count"] = word_count(txt)
    return dic

def evaluation(true_labels, predicted_labels):
    """Uses true labels and predicted labels to calculate precision/recall/f1 and a confusion matrix for each class."""
    # Print classification report
    print(classification_report(true_labels, predicted_labels, digits=3))
    # Construct a confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels,
                          labels=['books', 'camera', 'dvd', 'health', 'music', 'software'])
    print(cm)
    # Print a visual representation of the CM
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=['books', 'camera', 'dvd', 'health', 'music', 'software'])
    disp.plot()
    return


if __name__ == "__main__":
    # Load SpaCy language model
    nlp = spacy.load('en_core_web_sm')

    # Generate train/dev/test split
    X_train, X_test, X_dev, y_train, y_test, y_dev = split_corpus()

    # Construct a set of features
    features = construct_features()

    # # Get naive bayes prediction and evaluate
    # naive_bayes_pred = naive_bayes(features)
    # evaluation(y_test, naive_bayes_pred)
    # naive_bayes_acc = accuracy_score(y_test, naive_bayes_pred)
    # print(f"Final accuracy naive bayes: {naive_bayes_acc}")
    #
    # # Get decision tree prediction and evaluate
    # decision_tree_pred = decision_tree(features)
    # evaluation(y_test, decision_tree_pred)
    # decision_tree_acc = accuracy_score(y_test, decision_tree_pred)
    # print(f"Final accuracy decision tree: {decision_tree_acc}")
    #
    # # Get random forest prediction and evaluate
    # random_forest_pred = random_forest(features)
    # evaluation(y_test, random_forest_pred)
    # random_forest_acc = accuracy_score(y_test, random_forest_pred)
    # print(f"Final accuracy random forest: {random_forest_acc}")
    #
    # # Get KNN prediction and evaluate
    # knn_pred = knn(features)
    # evaluation(y_test, knn_pred)
    # knn_acc = accuracy_score(y_test, knn_pred)
    # print(f"Final accuracy KNN: {knn_acc}")
    #
    # # Get SVM prediction and evaluate
    # svm_pred = supvecmac(features)
    # evaluation(y_test, svm_pred)
    # svm_acc = accuracy_score(y_test, svm_pred)
    # print(f"Final accuracy svm: {svm_acc}")

    # Get linear SVM prediction and evaluate
    linear_svm_pred = linear_svm(features)
    evaluation(y_test, linear_svm_pred)
    linear_svm_acc = accuracy_score(y_test, linear_svm_pred)
    print(f"Final accuracy linear SVM: {linear_svm_acc}")
