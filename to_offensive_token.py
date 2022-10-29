import argparse
import pandas as pd
import csv

def offensive(text):
    with open(args.offensive_data, newline='') as f:
        reader = csv.reader(f)
        offensive_list = list(reader)
    offensive_list = [item for sublist in offensive_list for item in sublist]
    if isinstance(text, str):
        words = text.split()
        new_sentence = ""
        for word in words:
            if word in offensive_list:
                word = 'OFFENSIVE'
            new_sentence = new_sentence + " " + word
        text = new_sentence
    return text



def offensive_tokenizer(df):
    df = df.apply(lambda x: offensive(x))
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-tf", "--train_file", default='preprocessed_data/train.csv', type=str,
                        help="Train file to learn from (default train.csv)")
    parser.add_argument("-df", "--dev_file", default='preprocessed_data/dev.csv', type=str,
                        help="Dev file to evaluate on (default dev.csv)")
    parser.add_argument("-tsf", "--test_file", default='preprocessed_data/test.csv', type=str,
                        help="Test file to evaluate on (default test.csv)")
    parser.add_argument("-od", "--offensive_data", default='data_offensive_words/luis_von_ahn_badwords.csv', type=str,
                        help="Offensive data to learn from (default luis_von_ahn_badwords.csv)")
    args = parser.parse_args()

    df_train = pd.read_csv(args.train_file, sep=',', names=['number', 'text', 'label'])
    df_dev = pd.read_csv(args.dev_file, sep=',', names=['number','text', 'label'])
    df_test = pd.read_csv(args.test_file, sep=',', names=['number', 'text', 'label'])

    print('hoi')
    # Preprocessing the data
    df_train['text'] = offensive_tokenizer(df_train['text'])
    df_test['text'] = offensive_tokenizer(df_test['text'])
    df_dev['text'] = offensive_tokenizer(df_dev['text'])

    df_train.to_csv('offensive_token_data/train.csv', encoding="utf-8")
    df_test.to_csv('offensive_token_data/test.csv', encoding="utf-8")
    df_dev.to_csv('offensive_token_data/dev.csv', encoding="utf-8")

