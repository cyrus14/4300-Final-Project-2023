import pandas as pd
import ast
import pickle
import numpy as np
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize

df = pd.read_csv('../backend/dataset/big_df_edited.csv')

df = df.drop(columns=['Unnamed: 0'])

df = df.loc[df['tag'] != 'misc']

df['emotions'] = df['emotions'].apply(ast.literal_eval)
if type(df['emotions'][0][0]) != str:
    df['emotions'] = df['emotions'].apply(lambda x: [tup[0] for tup in x])

print(set([j for sub in df['emotions'].values for j in sub]))
bad_tags = {'aggression': 'aggressive', 'anger': 'angry', 'angst-ridden': 'angst', 'anxiety': 'anxious',
            'calmed': 'calm', 'calmness': 'calm', 'celebrate': 'celebratory', 'cheer': 'cheerful',
            'cheer-up': 'cheerful', 'cheering': 'cheerful', 'cheerup': 'cheerful', 'contemplate': 'contemplative',
            'cry': 'crying', 'depressed': 'depressing', 'depressive': 'depressing', 'excite': 'excited',
            'excitement': 'excited', 'exciting': 'excited', 'fight': 'fighting', 'funereal': 'sad',
            'happiness': 'happy', 'happy songs': 'happy', 'happy music': 'happy',
            'heartbreaking': 'heartbroken', 'heartbreak': 'heartbroken', 'humor': 'humorous',
            'intensive': 'intense', 'joyous': 'joyful',
            'lament': 'lamenting', 'melancholy': 'melancholic', 'misery': 'miserable', 'mourn': 'mournful',
            'outrage': 'outrageous', 'passion': 'passionate', 'peace': 'peaceful', 'quietly': 'quiet',
            'quietness': 'quiet', 'rebel': 'rebellious', 'relax': 'relaxed', 'relaxing': 'relaxed',
            'sadness': 'sad', 'sob': 'sobbing', 'soothe': 'soothing', 'thrill': 'thrilling',
            'tragedy': 'tragic', 'vigor': 'vigorous', 'weep': 'weeping'}

df['emotions'] = df['emotions'].apply(
    lambda ls: [bad_tags[tag] if tag in bad_tags else tag for tag in ls])

df['log_views'] = np.log(df['views'] + 1)
df['norm_views'] = df['log_views'] / max(df['log_views'])

unique_tags = set([j for sub in df['emotions'].values for j in sub])
print(unique_tags)

with open('../backend/unique_tags.pkl', 'wb') as f:
    pickle.dump(unique_tags, f)

tag_to_index = {t: i for i, t in enumerate(unique_tags)}
index_to_tag = {i: t for i, t in enumerate(unique_tags)}

with open('../backend/tag_to_index.pkl', 'wb') as f:
    pickle.dump(tag_to_index, f)
with open('../backend/index_to_tag.pkl', 'wb') as f:
    pickle.dump(index_to_tag, f)

song_tag_mat = np.zeros((df.shape[0], len(unique_tags)))
for i in range(df.shape[0]):
    tags = df['emotions'].iloc[i]
    for t in tags:
        j = tag_to_index[t]
        song_tag_mat[i, j] = 1


N_FEATS = 40
docs_compressed, s, words_compressed = svds(song_tag_mat, k=N_FEATS)
words_compressed = words_compressed.transpose()
words_compressed_normed = normalize(words_compressed, axis=1)
docs_compressed_normed = normalize(docs_compressed, axis=1)


td_matrix_np = song_tag_mat.transpose()
td_matrix_np = normalize(td_matrix_np)

with open('../backend/words_compressed.pkl', 'wb') as f:
    pickle.dump(words_compressed, f)

with open('../backend/docs_compressed_normed.pkl', 'wb') as f:
    pickle.dump(docs_compressed_normed, f)

df.to_csv('../backend/big_df_edited.csv')
