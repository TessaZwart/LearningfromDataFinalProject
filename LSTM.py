import nltk
import pandas as pd
import numpy as np
import tf as tensorflow
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

    lstm_units = 512
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

def test_set_predict(model, X_test, Y_test):
    # Get predictions using the trained model
    Y_pred = model.predict(X_test)
    # Finally, convert to numerical labels to get scores with sklearn
    Y_pred = np.argmax(Y_pred, axis=1)
    # If you have gold data, you can calculate accuracy
    Y_test = np.argmax(Y_test, axis=1)

    return Y_test, Y_pred
    

def main():

    args = create_arg_parser()
    x_train_padded, x_dev_padded, x_test_padded, Y_train_bin, Y_dev_bin, Y_test_bin, tokenizer = read_data(args)

    embedding_vectors = get_glove_embedding_vectors('glove.840B.300d.txt')

    embedding_matrix = get_embedding_matrix(embedding_vectors, tokenizer)

    lr = 0.001
    e = 2
    bs = 100
    drop=0.3
    
    model = create_model(embedding_matrix, tokenizer, lr, drop)
    model = train_model(model, x_train_padded, Y_train_bin, x_dev_padded, Y_dev_bin, e, bs)
    Y_test, Y_pred = test_set_predict(model, x_test_padded, Y_test_bin)
    print(evaluate_model(model, Y_test, Y_pred))
                    

if __name__ == '__main__':
    main()
