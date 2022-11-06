import nltk
import pandas as pd
import numpy as np
import tensorflow as tf
import keras
import argparse

from evaluation import evaluate_model

from sklearn import preprocessing
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score, f1_score

from tensorflow.python.keras.layers import LSTM, Dense, Embedding, Dropout
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras import layers
from tensorflow.python.keras.optimizers import adam_v2

def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-tf", "--train_file", default='preprocessed_data/train.csv', type=str,
                        help="Train file to learn from (default preprocessed_data/train.csv)")
    parser.add_argument("-df", "--dev_file", default='preprocessed_data/dev.csv', type=str,
                        help="Dev file to evaluate on (default preprocessed_data/dev.csv)")
    parser.add_argument("-tef", "--test_file", default='preprocessed_data/test.csv', type=str,
                        help="Test file to evaluate on (default preprocessed_data/test.csv)")
    args = parser.parse_args()
    print('parse done')
    return args

def read_data(args):
    use_tokens = False

    # Loading dataframes
    df_train = pd.read_csv(args.train_file, sep=',', names=['text', 'label'])
    df_dev = pd.read_csv(args.dev_file, sep=',', names=['text', 'label'])
    df_test = pd.read_csv(args.test_file, sep=',', names=['text', 'label'])

    # Converting comments to strings
    df_train['text'] = df_train['text'].astype(str)
    df_dev['text'] = df_dev['text'].astype(str)
    df_test['text'] = df_test['text'].astype(str)

    # Deleting empty data
    df_train.dropna()
    df_dev.dropna()
    df_test.dropna()

    # Loading dataframes into variables
    x_train, y_train, x_dev, y_dev, x_test, y_test = df_train['text'], df_train['label'], df_dev['text'], df_dev['label'], df_test['text'], df_test['label']

    # Transform string labels to one-hot encodings
    encoder = LabelBinarizer()

    Y_train_bin = encoder.fit_transform(y_train)
    Y_dev_bin = encoder.fit_transform(y_dev)
    Y_test_bin = encoder.fit_transform(y_test)

    # Tokenize the data
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(df_train['text'])
    tokenizer.fit_on_texts(df_dev['text'])
    tokenizer.fit_on_texts(df_test['text'])

    x_train_sequences = tokenizer.texts_to_sequences(df_train['text'])
    x_development_sequences = tokenizer.texts_to_sequences(df_dev['text'])
    x_test_sequences = tokenizer.texts_to_sequences(df_test['text'])


    # Pad the data
    max_comment_length = 300
    x_train_padded = tf.keras.preprocessing.sequence.pad_sequences(x_train_sequences, maxlen=max_comment_length)
    x_dev_padded = tf.keras.preprocessing.sequence.pad_sequences(x_development_sequences, maxlen=max_comment_length)
    x_test_padded = tf.keras.preprocessing.sequence.pad_sequences(x_test_sequences, maxlen=max_comment_length)

    return x_train_padded, x_dev_padded, x_test_padded, Y_train_bin, Y_dev_bin, Y_test_bin, tokenizer

def get_glove_embedding_vectors(glove_filename):
    embedding_vectors = {}

    with open(glove_filename,'r',encoding='utf-8') as file:
        for row in file:
            values = row.split(' ')
            word = values[0]
            weights = np.asarray([float(value) for value in values[1:]])
            embedding_vectors[word] = weights
    return embedding_vectors


def get_embedding_matrix(embedding_vectors, tokenizer):
    embedding_dim = 300
    vocab_length = len(tokenizer.word_index)+1

    embedding_matrix = np.zeros((vocab_length, embedding_dim))
    for word, index in tokenizer.word_index.items():
        if index < vocab_length:
            embedding_vector = embedding_vectors.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector
    return embedding_matrix


def create_model(embedding_matrix, tokenizer, lr, dropout):
    vocab_length = len(tokenizer.word_index)+1
    embedding_dim = 300

    lstm_units = 1000
    opt = adam_v2.Adam(learning_rate=lr)


    model = Sequential()
    model.add(Embedding(vocab_length, embedding_dim, trainable = False, weights=[embedding_matrix]))
    model.add(LSTM(lstm_units, return_sequences=True))
    model.add(LSTM(lstm_units))
    model.add(Dropout(dropout))
    model.add(layers.Dense(1, activation='softmax' ))
    model.compile(loss='binary_crossentropy',optimizer=opt, metrics=['accuracy'])
    model.summary()

    return model

def train_model(model, x_train_padded, Y_train_bin, x_dev_padded, Y_dev_bin, epochs, batch_size):

    model.fit(pd.DataFrame(x_train_padded), Y_train_bin,
                    epochs=epochs,
                    verbose=False,
                    shuffle=True,
                    validation_data=(pd.DataFrame(x_dev_padded), Y_dev_bin),
                    batch_size=batch_size,
                    )
    return model



def evaluate(model, x_test_padded, Y_test_bin):
    metrics = model.evaluate(pd.DataFrame(x_test_padded), Y_test_bin)
    predictions = model.predict_classes(pd.DataFrame(x_test_padded))
    labels = Y_test_bin.argmax(axis=1)

    report = classification_report(labels , predictions,output_dict=True, digits=3)
    report = pd.DataFrame(report).transpose()
    print(report.to_latex())


def test_set_predict(model, X_test, Y_test):
    ''' blablabla'''
    # Get predictions using the trained model
    Y_pred = model.predict(X_test)
    # Finally, convert to numerical labels to get scores with sklearn
    Y_pred = np.argmax(Y_pred, axis=1)
    # If you have gold data, you can calculate accuracy
    Y_test = np.argmax(Y_test, axis=1)

    print("extra")
    print('Accuracy on own {1} set: {0}'.format(round(accuracy_score(Y_test, Y_pred), 5), ident))
    print("(macro) F1 score of test set:")
    print(f1_score(Y_test, Y_pred, average='macro'))
    print("extra^")
    return Y_test, Y_pred

def main():

    args = create_arg_parser()
    x_train_padded, x_dev_padded, x_test_padded, Y_train_bin, Y_dev_bin, Y_test_bin, tokenizer = read_data(args)

    print('----------Matrix generation-----------------')

    embedding_vectors = get_glove_embedding_vectors('glove.840B.300d.txt')

    embedding_matrix = get_embedding_matrix(embedding_vectors, tokenizer)

    print('----------Create model-----------------')

    learning_rates = [0.01, 0.001, 0.0001]
    epochs = [50, 75, 100]
    batch_sizes = [16, 32, 64, 128]
    dropouts = [0.0, 0.3, 0.5, 0.7]

    for lr in learning_rates:
        for e in epochs:
            for bs in batch_sizes:
                for drop in dropouts:
                    print('-------------------New predictions------------------------')
                    print('Learning rate: ', lr)
                    print('Number of epochs: ', e)
                    print('Batch size: ', bs)
                    print('Dropout: ', drop)
                    model = create_model(embedding_matrix, tokenizer, lr, drop)

                    print('----------Training-----------------')

                    model = train_model(model, x_train_padded, Y_train_bin, x_dev_padded, Y_dev_bin, e, bs)

                    # print('----------Evaluation-----------------')
                    #
                    # evaluate(model, x_test_padded, Y_test_bin)
                    #
                    print('----------Prediction-----------------')
                    #
                    Y_test, Y_pred = test_set_predict(model, x_test_padded, Y_test_bin)

                    print('---------------Evaluate function--------------')

                    print(evaluate_model(Y_test, Y_pred))
                    #print(evaluate_model(Y_test_bin, Y_test_bin))

if __name__ == '__main__':
    main()
