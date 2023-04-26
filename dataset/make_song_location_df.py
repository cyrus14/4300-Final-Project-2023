import pandas as pd
import numpy as np
import re

import wiki_scraping
from  wiki_scraping import wiki_articles_to_include

lyric_df = pd.read_csv('lyrics_tokenizer/shortened_tokenized_lyrics.csv')
lyric_df.drop(columns=['Unnamed: 0', 'lyrics'],inplace=True)

lyr_loc_df = np.zeros((len(lyric_df), len(wiki_articles_to_include.titles)))

def tokenize(text): #taken from CS 4300
    return [x for x in re.findall(r"[a-z]+", text.lower()) if x != ""]

def find_in_song(lyrics, loc_name):
    for s in tokenize(loc_name):
        if s not in lyrics:
            return 0
    return 1

for i, location in enumerate(wiki_articles_to_include.titles):
    lyr_loc_df[:, i] = lyric_df['tknzd_lyrics'].apply(lambda x: find_in_song(x, location))


location_to_index = {l:i for i, l in enumerate(wiki_articles_to_include.titles)}
index_to_location = {i:l for i, l in enumerate(wiki_articles_to_include.titles)}

song_to_index = {s:i for i, s in enumerate(lyric_df['title'])}
index_to_song = {i:s for i, s in enumerate(lyric_df['title'])}


def search(query_city, df):
    c_i = location_to_index[query_city]
    sngs = lyr_loc_df[:,c_i]
    results = pd.DataFrame()
    for i in  np.where(sngs==1)[0]:
        #song_title = index_to_song[i]
        results = results.append(lyric_df.iloc[i])
        results = results.sort_values(by='views', ascending=False)
    return results[:10].reset_index(drop=True)

print("New York City", search('New York City', lyr_loc_df))