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

# from app import wiki_tfidf, song_tfidf, loc_to_idx, song_to_idx, idx_to_song, big_df

# PICKLE :)


root_path = os.path.abspath(os.curdir)
os.environ['ROOT_PATH'] = root_path


# unpickle wiki_tf_idf (vec2)
with open(os.environ['ROOT_PATH']  + '/wiki_tf_idf.pkl', 'rb') as pickle_file:
    wiki_tfidf = pickle.load(pickle_file)#.toarray()

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

with open(os.environ['ROOT_PATH'] + '/vectorizer.pkl', 'rb') as pickle_file:
    vectorizer = pickle.load(pickle_file)

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

with zipfile.ZipFile(os.environ['ROOT_PATH'] + '/dataset/big_df_edited.csv.zip', 'r') as zip_ref:
    zip_ref.extractall(os.environ['ROOT_PATH'] + '/dataset/')

# read in "edited" csv (shortened)
# items = os.listdir(root_path)

# for item in items:
#     print(item)


big_df = pd.read_csv(os.environ['ROOT_PATH'] + '/dataset/big_df_edited.csv')

big_df['emotions'] = big_df['emotions'].apply(ast.literal_eval)
big_df['emotions'] = big_df['emotions'].apply(lambda x: [tup[0] for tup in x])

# unique_tags = set([j for sub in big_df['emotions'].values for j in sub])
# tag_to_index = {t:i for i, t in enumerate(unique_tags)}
# index_to_tag = {i:t for i, t in enumerate(unique_tags)}

# song_tag_mat = np.zeros((big_df.shape[0], len(unique_tags)))
# for i in range(big_df.shape[0]):
#     tags = big_df['emotions'].iloc[i]
#     for t in tags:
#         j = tag_to_index[t]
#         song_tag_mat[i, j] = 1

# big_df['log_views'] = np.log(big_df['views'] + 1)
# big_df['norm_views'] = big_df['log_views'] / max(big_df['log_views'])

# N_FEATS = 10
# docs_compressed, s, words_compressed = svds(song_tag_mat, k=N_FEATS)
# words_compressed = words_compressed.transpose()
# words_compressed_normed = normalize(words_compressed, axis = 1)
# docs_compressed_normed = normalize(docs_compressed, axis=1)

# td_matrix_np = song_tag_mat.transpose()
# td_matrix_np = normalize(td_matrix_np)

def get_query_vec(query):
    query_tagf = np.zeros((1,len(unique_tags)))
    for wrd in query.split():
        if wrd in unique_tags:
            i = tag_to_index[wrd]
            query_tagf[0,i] += 1
    return normalize(query_tagf @ words_compressed)

def closest_songs_to_query(query, k = 5):
    query_vec = get_query_vec(query)
    sims = normalize(query_vec).dot(docs_compressed_normed.T)[0]
    asort = np.argsort(-sims)[:k+1]
    return [ {'title':big_df['title'].iloc[i], 
              'artist':big_df['artist'].iloc[i], 
              'year':big_df['year'].iloc[i], 
              'views':big_df['views'].iloc[i], 
              'sim':sims[i], 
              'score':sims[i]} for i in asort[0:]]


'''
# unpickle wiki_tf_idf (vec2)
with open('wiki_tf_idf.pkl', 'rb') as pickle_file:
    wiki_tfidf = pickle.load(pickle_file)

# unpickle song_tf_idf (X)
with open('song_tf_idf.pkl', 'rb') as pickle_file:
    song_tfidf = pickle.load(pickle_file)

# unpickle loc_to_index
with open('loc_to_index.pkl', 'rb') as pickle_file:
    loc_to_idx = pickle.load(pickle_file)

# unpickle song_to_index
with open('song_to_index.pkl', 'rb') as pickle_file:
    song_to_idx = pickle.load(pickle_file)

# unpickle index_to_song
with open('index_to_song.pkl', 'rb') as pickle_file:
    idx_to_song = pickle.load(pickle_file)

# read in "edited" csv (shortened)
big_df = pd.read_csv('dataset/big_df_edited.csv')
'''

def test():
    print('hello')

# cosine similarity function
def cos_sim(city, song):
    city_i = loc_to_idx[city]
    song_i = song_to_idx[song]
    city_vec = wiki_tfidf[city_i, :]
    song_vec = song_tfidf[song_i, :]
    denom = np.linalg.norm(city_vec) * np.linalg.norm(song_vec)
    num = city_vec @ song_vec
    return (num + 0.5) / (denom + 0.5)

def top_songs_query(city, query = "sad energetic"):
    best = []
    returned = []
    # query_emot_vec = closest_songs_to_query(query, k=10)

    query_vec = get_query_vec(query)
    for song in song_to_idx:
        sim = cos_sim(city, song)

        if sim == 1.0:
            sim = 0
    
        pop = big_df['norm_views'].iloc[song_to_idx[song]] 

        song_emot_vec = docs_compressed_normed[song_to_idx[song], :]
        emot_score = np.exp(query_vec @ song_emot_vec)/np.e
        # score = (sim ** 2) + (pop / 5) + ((emot_score) / 10)
        score = (sim + 1) * (pop + 1) * (emot_score + 1) / 6

        best.append((song, sim, pop, emot_score, score))
    srtd = sorted(best, key=lambda x: x[-1], reverse=True)
    for t in srtd[:10]:
        retrieved = big_df.iloc[song_to_idx[t[0]]]
        result = {'title': retrieved['title'],
                  'artist': retrieved['artist'],
                  'year': retrieved['year'],
                  'views': retrieved['views'],
                  'sim': t[1],
                  'pop':t[2],
                  'emot': t[3],
                  'score': t[-1]}
        prod = song_tfidf[song_to_idx[retrieved['title']]] * wiki_tfidf[loc_to_idx[city]]
        strongest = np.argsort(prod)[-10:]
        strongest_words = [index_to_word[w] for w in strongest]
        print(retrieved['title'])
        print(strongest_words)
        result['best_words'] = strongest_words
        result['score_in_song'] = song_tfidf[song_to_idx[retrieved['title']],strongest]
        result['score_in_city'] = wiki_tfidf[loc_to_idx[city],strongest]
        returned.append(result)

    return returned

# print(top_songs_query("New York City"))
# print(list(unique_tags))
