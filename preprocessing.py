import nltk
import re

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
    text = re.sub(r'\d+', " ", text)  # Remove numbers
    text = re.sub(r'#', ' ', text)  # remove hashtags
    #text = re.sub(r"[^a-zA-Z]", " ", text)  # Removes special chars
    text = text.lower()
    text = text.split()
    text = " ".join([word for word in text if not word in stop_word])  # Remove stop words
    # text = perform_stemming(text)

    return text


def clean_dataframe(df):
    df = df.apply(lambda x: clean(x))
    return df


def data_vectorizer(x_train, y_train, x_test, y_test):
    le = LabelEncoder()
    le.fit(y_train)
    y_train = le.transform(y_train)

    vec = CountVectorizer()
    vec.fit(x_train, x_test)
    x_train = vec.transform(x_train)
    x_test = vec.transform(x_test)

    return x_train, y_train, x_test, y_test, le
