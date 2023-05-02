import pickle
import pandas as pd
import numpy as np
import json
import math
import os
import zipfile
import ast
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize
import random
from collections import Counter
import functools

# PICKLE :)

root_path = os.path.abspath(os.curdir)
os.environ['ROOT_PATH'] = root_path


# unpickle wiki_tf_idf (vec2)
with open(os.environ['ROOT_PATH'] + '/wiki_tf_idf.pkl', 'rb') as pickle_file:
    wiki_tfidf = pickle.load(pickle_file)  # .toarray()

# unpickle song_tf_idf (X)
with open(os.environ['ROOT_PATH'] + '/song_tf_idf.pkl', 'rb') as pickle_file:
    song_tfidf = pickle.load(pickle_file).toarray()

# unpickle loc_to_index
with open(os.environ['ROOT_PATH'] + '/loc_to_index.pkl', 'rb') as pickle_file:
    loc_to_idx = pickle.load(pickle_file)

# unpickle song_to_index
with open(os.environ['ROOT_PATH'] + '/song_to_index.pkl', 'rb') as pickle_file:
    song_to_idx = pickle.load(pickle_file)

# unpickle index_to_song
with open(os.environ['ROOT_PATH'] + '/index_to_song.pkl', 'rb') as pickle_file:
    idx_to_song = pickle.load(pickle_file)

with open(os.environ['ROOT_PATH'] + '/unique_tags.pkl', 'rb') as pickle_file:
    unique_tags = pickle.load(pickle_file)

with open(os.environ['ROOT_PATH'] + '/tag_to_index.pkl', 'rb') as pickle_file:
    tag_to_index = pickle.load(pickle_file)

with open(os.environ['ROOT_PATH'] + '/index_to_tag.pkl', 'rb') as pickle_file:
    index_to_tag = pickle.load(pickle_file)

with open(os.environ['ROOT_PATH'] + '/index_to_word.pkl', 'rb') as pickle_file:
    index_to_word = pickle.load(pickle_file)

with open(os.environ['ROOT_PATH'] + '/word_to_index.pkl', 'rb') as pickle_file:
    word_to_index = pickle.load(pickle_file)

with open(os.environ['ROOT_PATH'] + '/words_compressed.pkl', 'rb') as pickle_file:
    words_compressed = pickle.load(pickle_file)

with open(os.environ['ROOT_PATH'] + '/docs_compressed_normed.pkl', 'rb') as pickle_file:
    docs_compressed_normed = pickle.load(pickle_file)

with open(os.environ['ROOT_PATH'] + '/song_inv_idx.pkl', 'rb') as pickle_file:
    song_inv_idx = pickle.load(pickle_file)

with open(os.environ['ROOT_PATH'] + '/idf.pkl', 'rb') as pickle_file:
    idf = pickle.load(pickle_file)

with open(os.environ['ROOT_PATH'] + '/norms.pkl', 'rb') as pickle_file:
    norms = pickle.load(pickle_file)

with zipfile.ZipFile(os.environ['ROOT_PATH'] + '/dataset/big_df_edited.csv.zip', 'r') as zip_ref:
    zip_ref.extractall(os.environ['ROOT_PATH'] + '/dataset/')

big_df = pd.read_csv(os.environ['ROOT_PATH'] + '/dataset/big_df_edited.csv')

big_df['emotions'] = big_df['emotions'].apply(ast.literal_eval)
big_df['emotions'] = big_df['emotions'].apply(lambda x: [tup[0] for tup in x])


@functools.lru_cache(maxsize=None)
def get_query_vec(query):
    query_tagf = np.zeros((1, len(unique_tags)))
    for wrd in query.split():
        if wrd in unique_tags:
            i = tag_to_index[wrd]
            query_tagf[0, i] += 1
    return normalize(query_tagf @ words_compressed)


@functools.lru_cache(maxsize=None)
def top_songs_query(city, query):
    """
    city name to query on
    query is "happy excited" for example
    """
    # print("\nSTART\n")
    best = []
    returned = []
    query_vec = get_query_vec(query)
    # print(wiki_tfidf.shape)
    # print(song_tfidf.shape)
    city_vec = wiki_tfidf[loc_to_idx[city], :]
    # print("city_vec, ", city_vec.shape)
    # np.dot(city_vec @ song_tfidf).sum(axis=1)
    lyr_sym = song_tfidf @ city_vec.T
    # print("lyr sim", lyr_sym.shape)
    # (query_vec @ docs_compressed_normed).sum(axis=1)
    emot_sym = (docs_compressed_normed@query_vec.T).squeeze()
    if emot_sym.max() > emot_sym.min():
        emot_sym = (emot_sym - emot_sym.min()) / \
            (emot_sym.max() - emot_sym.min())
    else:
        emot_sym = np.zeros(len(emot_sym))
    # print("emot_sym", emot_sym.shape)

    alpha = .7
    beta = .3

    score = alpha * lyr_sym + beta * emot_sym
    # print("score", score.shape)
    best_songs = np.argsort(-score)[:15]
    for i, ind in enumerate(best_songs):
        song = idx_to_song[ind]
        # print(song)
        pop = big_df['norm_views'].iloc[ind]
        best.append((song, lyr_sym[ind], pop, emot_sym[ind], score[ind]))
    # print('here')
    for t in best:
        retrieved = big_df.iloc[song_to_idx[t[0]]]
        result = {'title': retrieved['title'],
                  'artist': retrieved['artist'],
                  'year': int(retrieved['year']),
                  'views': int(retrieved['views']),
                  'sim':   float(t[1]),
                  'pop': float(t[2]),
                  'emot': float(t[3]),
                  'score': float(t[-1]),
                  'id': int(song_to_idx[retrieved['title']])}
        prod = song_tfidf[song_to_idx[retrieved['title']]] * \
            wiki_tfidf[loc_to_idx[city]]
        strongest = np.argsort(prod)[-10:]
        strongest_words = [index_to_word[w] for w in strongest]
        # print(retrieved['title'])
        # print(strongest_words)
        result['best_words'] = strongest_words
        result['score_in_song'] = list(
            song_tfidf[song_to_idx[retrieved['title']], strongest])
        result['score_in_city'] = list(wiki_tfidf[loc_to_idx[city], strongest])

        # print(result['score_in_song'], type(result['score_in_song']))
        # print(result['score_in_city'], type(result['score_in_city']))
        # print(result['id'], type(result['id']))

        returned.append(result)
    print("END QUERY")
    return returned
