import nltk
import re
import argparse
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import *

# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('omw-1.4')

stop_word = stopwords.words('english')

def perform_stemming(text):
    stemmer = PorterStemmer()
    stemmed_words = [stemmer.stem(word) for word in text.split()]
    stemmed_sentence = ' '.join(stemmed_words)
    return stemmed_sentence


def clean(text):
    text = re.sub(r'URL', "", text)  # Remove URL's
    text = re.sub(r'@USER', "", text)  # Remove @users

    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"\'scuse", "excuse", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"can't", "cannot", text)

    text = re.sub(r'\d+', "", text)  # Remove numbers
    text = re.sub(r'#', '', text)  # remove hashtags
    text = re.sub(r"[^a-zA-Z]", " ", text)  # Removes special chars
    text = text.lower()
    text = text.split()
    text = " ".join([word for word in text if not word in stop_word])  # Remove stop words
    # text = perform_stemming(text)

    return text


def clean_dataframe(df):
    df = df.apply(lambda x: clean(x))
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-tf", "--train_file", default='data/train.tsv', type=str,
                        help="Train file to learn from (default train.tsv)")
    parser.add_argument("-df", "--dev_file", default='data/dev.tsv', type=str,
                        help="Dev file to evaluate on (default dev.tsv)")
    parser.add_argument("-tsf", "--test_file", default='data/test.tsv', type=str,
                        help="Test file to evaluate on (default test.tsv)")
    args = parser.parse_args()

    df_train = pd.read_csv(args.train_file, sep='\t', names=['text', 'label'])
    df_dev = pd.read_csv(args.dev_file, sep='\t', names=['text', 'label'])
    df_test = pd.read_csv(args.test_file, sep='\t', names=['text', 'label'])
    print('hoi')
    # Preprocessing the data
    df_train['text'] = clean_dataframe(df_train['text'])
    df_test['text'] = clean_dataframe(df_test['text'])
    df_dev['text'] = clean_dataframe(df_dev['text'])

    df_train.to_csv('preprocessed_data/train.csv', encoding="utf-8")
    df_test.to_csv('preprocessed_data/test.csv', encoding="utf-8")
    df_dev.to_csv('preprocessed_data/dev.csv', encoding="utf-8")
