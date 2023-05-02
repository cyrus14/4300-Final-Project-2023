import numpy as np
import pickle
import os
import json
from collections import Counter
import pandas as pd
import ast

big_df = pd.read_csv('dataset/big_df_edited.csv')
big_df['tknzd_lyrics'] = big_df['tknzd_lyrics'].apply(ast.literal_eval)
big_df['emotions'] = big_df['emotions'].apply(ast.literal_eval)
big_df['social_tags'] = big_df['social_tags'].apply(ast.literal_eval)


root_path = os.path.abspath(os.curdir)
os.environ['ROOT_PATH'] = root_path

with open(os.environ['ROOT_PATH'] + '/wiki_tf_idf.pkl', 'rb') as pickle_file:
    wiki_tfidf = pickle.load(pickle_file)  # .toarray()

with open(os.environ['ROOT_PATH'] + '/song_tf_idf.pkl', 'rb') as pickle_file:
    song_tfidf = pickle.load(pickle_file).toarray()

f = open('dataset/wiki_scraping/wiki_texts.json')
wiki_texts = json.load(f)

inv_idx = {}
for i, song in enumerate(big_df['tknzd_lyrics']):
    cntr = Counter(song)
    seen = {}
    for wrd in cntr:
        if wrd in seen:
            seen[wrd] += 1
        else:
            seen[wrd] = 1
    for wrd in seen:
        tup = (i, seen[wrd])
        if wrd in inv_idx:
            inv_idx[wrd].append(tup)
        else:
            inv_idx[wrd] = [tup]
with open('song_inv_idx.pkl', 'wb') as f:
    pickle.dump(inv_idx, f)

idf = {}
n_docs = big_df.shape[0]
max_df = .8 * n_docs
min_df = 10
for wrd in inv_idx:
    n_docs_wrd = len(inv_idx[wrd])
    # if n_docs_wrd >= min_df and n_docs_wrd < max_df:
    idf_t = np.log2(n_docs / (1 + n_docs_wrd))
    idf[wrd] = idf_t

with open('idf.pkl', 'wb') as f:
    pickle.dump(idf, f)

norms = np.zeros(n_docs)
for wrd in idf:
    if wrd in inv_idx:
        for docid, count in inv_idx[wrd]:
            norms[docid] += (count * idf[wrd]) ** 2
norms = np.sqrt(norms)

with open('norms.pkl', 'wb') as f:
    pickle.dump(norms, f)


# pickle all, do rest in query.py
