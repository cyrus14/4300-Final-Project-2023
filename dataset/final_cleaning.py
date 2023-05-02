import pandas as pd
import ast
import pickle
import numpy as np
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize

# MOVE ALL PKL FILES TO BACKEND!

df = pd.read_csv('../backend/dataset/big_df_edited.csv')

df = df.drop(columns=['Unnamed: 0'])

df = df.loc[df['tag'] != 'misc']

df['emotions'] = df['emotions'].apply(ast.literal_eval)
if type(df['emotions'][0][0]) != str:
    df['emotions'] = df['emotions'].apply(lambda x: [tup[0] for tup in x])

df['log_views'] = np.log(df['views'] + 1)
df['norm_views'] = df['log_views'] / max(df['log_views'])

unique_tags = set([j for sub in df['emotions'].values for j in sub])

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
