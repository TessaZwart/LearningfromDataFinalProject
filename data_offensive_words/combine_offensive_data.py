import pandas as pd

bad_words_df_1 = pd.read_csv("bad-bad-words.csv", names=['offensive_word'])

bad_words_df_2 = pd.read_csv("luis_von_ahn_badwords.csv", names=['offensive_word'])

combined_words = pd.concat([bad_words_df_1,bad_words_df_2]).drop_duplicates().reset_index(drop=True)
print()