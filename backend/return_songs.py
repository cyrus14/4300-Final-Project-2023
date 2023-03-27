import pandas as pd
import numpy as np


def find_nonzero_indices(city):
    res = []
    song_location = pd.read_csv('./songs_locations_df.csv')
    x = song_location[song_location[city] == 1]
    x = x.sort_values(by = 'Views', ascending = False)[:10]
    x = x[['Title','Artist', 'Year']]
    return x.values.tolist()
    
if __name__ == "__main__":
    print(find_nonzero_indices("London"))