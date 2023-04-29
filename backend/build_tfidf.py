import json
import pandas as pd
from collections import Counter
import ast
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
from dataset import wiki_scraping

my_stop_words = text.ENGLISH_STOP_WORDS.union({'city'})


f = open('dataset/wiki_scraping/wiki_texts.json')
wiki_texts = json.load(f)

big_df = pd.read_csv('dataset/big_df_edited.csv')
big_df.drop(columns=['Unnamed: 0'], inplace=True)
big_df['tknzd_lyrics'] = big_df['tknzd_lyrics'].apply(ast.literal_eval)
big_df['emotions'] = big_df['emotions'].apply(ast.literal_eval)
big_df['social_tags'] = big_df['social_tags'].apply(ast.literal_eval)

drop_rows = []
for row in big_df.iterrows():
    if row[1]['emotions'] == []:
        drop_rows.append(row[0])
big_df.drop(drop_rows, inplace=True)

vectorizer = TfidfVectorizer(max_df = 0.8, min_df = 10, norm='l2', 
                             stop_words = list(my_stop_words))
X = vectorizer.fit_transform(big_df['tknzd_lyrics'].apply(lambda x: " ".join(x)))

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

index_to_word = {i:v for i, v in enumerate(vectorizer.get_feature_names_out())}
word_to_index = {v:i for i, v in enumerate(vectorizer.get_feature_names_out())}

with open('word_to_index.pkl', 'wb') as f:
    pickle.dump(word_to_index, f)
with open('index_to_word.pkl', 'wb') as f:
    pickle.dump(index_to_word, f)
with open('song_tf_idf.pkl', 'wb') as f:
    pickle.dump(X, f)

song_to_index = {s:i for i, s in enumerate(big_df['title'])}
index_to_song = {i:s for i, s in enumerate(big_df['title'])}

with open('song_to_index.pkl', 'wb') as f:
    pickle.dump(song_to_index, f)
with open('index_to_song.pkl', 'wb') as f:
    pickle.dump(index_to_song, f)

### WIKI

wiki_corpus = []
for ls in list(wiki_texts.values()):
    wiki_corpus.append(" ".join(ls))

vec2 = vectorizer.transform(wiki_corpus)
vec2 = vec2.toarray()
X = X.toarray()

loc_to_index = {cty:i for i, cty in enumerate(wiki_texts.keys())}
with open('wiki_tf_idf.pkl', 'wb') as f:
    pickle.dump(vec2, f)

with open('loc_to_index.pkl', 'wb') as f:
    pickle.dump(loc_to_index, f)