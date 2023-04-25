import pickle
import pandas as pd
import numpy as np
import json
import math
import os
import zipfile
# from app import wiki_tfidf, song_tfidf, loc_to_idx, song_to_idx, idx_to_song, big_df

# PICKLE :)

root_path = os.path.abspath(os.curdir)
os.environ['ROOT_PATH'] = root_path

# Get a list of all items in the current directory
items = os.listdir(root_path)

# Print the directories
print("Directories that you can cd into:")
for item in items:
    print(item)

# unpickle wiki_tf_idf (vec2)
with open(os.environ['ROOT_PATH'] + '/wiki_tf_idf.pkl', 'rb') as pickle_file:
    wiki_tfidf = pickle.load(pickle_file)

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

with zipfile.ZipFile(os.environ['ROOT_PATH'] + 'dataset/big_df_edited.csv.zip', 'r') as zip_ref:
    zip_ref.extractall(os.environ['ROOT_PATH'] + 'dataset/')

# read in "edited" csv (shortened)
big_df = pd.read_csv('dataset/big_df_edited.csv')

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
    return (num) / (denom)


def top_songs_query(city):
    best = []
    returned = []
    for song in song_to_idx:
        sim = cos_sim(city, song)
        pop = math.log(big_df.iloc[song_to_idx[song]]['views'] + 1)
        score = sim * pop

        best.append((song, sim, pop, score))
    srtd = sorted(best, key=lambda x: x[3], reverse=True)
    for t in srtd[:10]:
        retrieved = big_df.iloc[song_to_idx[t[0]]]
        result = {'title': retrieved['title'],
                  'artist': retrieved['artist'],
                  'year': retrieved['year'],
                  'views': retrieved['views'],
                  'sim': t[1],
                  'score': t[3]}

        returned.append(result)

    return returned

# top_songs_query("New York City")
