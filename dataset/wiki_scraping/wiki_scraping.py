{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/cyrus14/4300-Final-Project-2023/blob/master/dataset/wiki_scraping/wiki_scraping.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 2,
      "metadata": {
        "id": "PpTskCj7HkA3"
      },
      "outputs": [],
      "source": [
        "import json\n",
        "import pandas as pd\n",
        "from collections import Counter\n",
        "import ast\n",
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "import pickle\n",
        "from sklearn.feature_extraction.text import TfidfVectorizer\n",
        "from sklearn.feature_extraction import text"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from google.colab import drive\n",
        "drive.mount('/content/drive')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "yKzRW8-yH8N0",
        "outputId": "800fc9c3-385c-4fae-fdcb-086634c242b2"
      },
      "execution_count": 3,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Mounted at /content/drive\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "%cd /content/drive/MyDrive/apollocate/4300-Final-Project-2023/"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "8dTX8z6nHzHi",
        "outputId": "ef7129c4-8364-48aa-9d19-7f1df7df6db4"
      },
      "execution_count": 4,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "/content/drive/MyDrive/apollocate/4300-Final-Project-2023\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 10,
      "metadata": {
        "id": "G8l2-qRMHkA5"
      },
      "outputs": [],
      "source": [
        "f = open('dataset/wiki_scraping/wiki_texts.json')\n",
        "wiki_texts = json.load(f)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 11,
      "metadata": {
        "id": "C_2T8pyuHkA5"
      },
      "outputs": [],
      "source": [
        "big_df = pd.read_csv('dataset/big_df.csv')"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 12,
      "metadata": {
        "id": "w0NMqAZPHkA6"
      },
      "outputs": [],
      "source": [
        "big_df.drop(columns=['Unnamed: 0'], inplace=True)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 13,
      "metadata": {
        "id": "-Sqbk-8PHkA6"
      },
      "outputs": [],
      "source": [
        "big_df['tknzd_lyrics'] = big_df['tknzd_lyrics'].apply(ast.literal_eval)\n",
        "big_df['emotions'] = big_df['emotions'].apply(ast.literal_eval)\n",
        "big_df['social_tags'] = big_df['social_tags'].apply(ast.literal_eval)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 14,
      "metadata": {
        "id": "PMMRrmDEHkA6"
      },
      "outputs": [],
      "source": [
        "drop_rows = []\n",
        "for row in big_df.iterrows():\n",
        "    if row[1]['emotions'] == []:\n",
        "        drop_rows.append(row[0])\n",
        "big_df.drop(drop_rows, inplace=True)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 15,
      "metadata": {
        "id": "AwilKJKUHkA6"
      },
      "outputs": [],
      "source": [
        "my_stop_words = text.ENGLISH_STOP_WORDS.union({'city'})\n",
        "vectorizer = TfidfVectorizer(max_df = 0.8, min_df = 10, norm='l2', stop_words = list(my_stop_words))\n",
        "X = vectorizer.fit_transform(big_df['tknzd_lyrics'].apply(lambda x: \" \".join(x)))"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 16,
      "metadata": {
        "id": "4OCZNzLdHkA6"
      },
      "outputs": [],
      "source": [
        "index_to_word = {i:v for i, v in enumerate(vectorizer.get_feature_names_out())}\n",
        "word_to_index = {v:i for i, v in enumerate(vectorizer.get_feature_names_out())}"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 17,
      "metadata": {
        "id": "wsOxHtGoHkA7"
      },
      "outputs": [],
      "source": [
        "with open('word_to_index.pkl', 'wb') as f:\n",
        "    pickle.dump(word_to_index, f)\n",
        "with open('index_to_word.pkl', 'wb') as f:\n",
        "    pickle.dump(index_to_word, f)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 18,
      "metadata": {
        "id": "REZHOJMuHkA7"
      },
      "outputs": [],
      "source": [
        "with open('song_tf_idf.pickle', 'wb') as f:\n",
        "    pickle.dump(X, f)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 19,
      "metadata": {
        "id": "CO2x8SSbHkA8"
      },
      "outputs": [],
      "source": [
        "song_to_index = {s:i for i, s in enumerate(big_df['title'])}\n",
        "index_to_song = {i:s for i, s in enumerate(big_df['title'])}"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 20,
      "metadata": {
        "id": "3IjdFFS6HkA8"
      },
      "outputs": [],
      "source": [
        "with open('song_to_index.pkl', 'wb') as f:\n",
        "    pickle.dump(song_to_index, f)\n",
        "with open('index_to_song.pkl', 'wb') as f:\n",
        "    pickle.dump(index_to_song, f)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "LmLArolrHkA8"
      },
      "source": [
        "# wiki"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 21,
      "metadata": {
        "id": "wgJvGb0EHkA9"
      },
      "outputs": [],
      "source": [
        "from dataset import wiki_scraping"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "print(len(wiki_texts['Toronto']))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "F5h5Ar7kONm4",
        "outputId": "9a72779f-1f95-4e4e-8423-87eda79e4b05"
      },
      "execution_count": 22,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "17414\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 23,
      "metadata": {
        "id": "T6iaemrWHkA9"
      },
      "outputs": [],
      "source": [
        "wiki_corpus = []\n",
        "for ls in list(wiki_texts.values()):\n",
        "    wiki_corpus.append(\" \".join(ls))"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 24,
      "metadata": {
        "id": "PsTVTt4AHkA-"
      },
      "outputs": [],
      "source": [
        "vec2 = vectorizer.transform(wiki_corpus)\n",
        "vec2 = vec2.toarray()\n",
        "X = X.toarray()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 25,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "2xCg95_fHkA-",
        "outputId": "8a57df38-a198-4b72-ab6a-3db543fbc731"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([12.7484876 , 21.72531696,  7.906622  , 17.22400158, 10.24045599,\n",
              "       19.59373312])"
            ]
          },
          "metadata": {},
          "execution_count": 25
        }
      ],
      "source": [
        "vec2.sum(axis=1)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 26,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "CNyWbowGHkA_",
        "outputId": "f2e9fcec-45ad-4c96-fea6-6bc665e431dd"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "(6, 13784)\n",
            "(34554, 13784)\n"
          ]
        }
      ],
      "source": [
        "print(vec2.shape)\n",
        "print(X.shape)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 27,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "OEup-83KHkA_",
        "outputId": "c5d48888-86d2-4c7a-ad69-a61a3dcaa44c"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([0., 0., 0., ..., 0., 0., 0.])"
            ]
          },
          "metadata": {},
          "execution_count": 27
        }
      ],
      "source": [
        "vec2[2,:]"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 28,
      "metadata": {
        "id": "q6CgclQ4HkBA"
      },
      "outputs": [],
      "source": [
        "loc_to_index = {cty:i for i, cty in enumerate(wiki_texts.keys())}"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 29,
      "metadata": {
        "id": "k1ZHkbaUHkBA"
      },
      "outputs": [],
      "source": [
        "def cos_sim(city, song):\n",
        "    city_i = loc_to_index[city]\n",
        "    song_i = song_to_index[song]\n",
        "    city_vec = vec2[city_i, :]\n",
        "    song_vec = X[song_i, :]\n",
        "    denom = np.linalg.norm(city_vec) * np.linalg.norm(song_vec)\n",
        "    num = city_vec @ song_vec\n",
        "    return (num ) /  (denom )"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 30,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "1aSmHFnuHkBA",
        "outputId": "99db59a4-93df-4003-f1f2-32dcab8c7aa5"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(13784,)"
            ]
          },
          "metadata": {},
          "execution_count": 30
        }
      ],
      "source": [
        "vec2[2,:].shape"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 31,
      "metadata": {
        "id": "2_VoyvHGHkBA"
      },
      "outputs": [],
      "source": [
        "def best_songs_for_city(city):\n",
        "    best = []\n",
        "    for song in song_to_index:\n",
        "        sim = cos_sim(city, song)\n",
        "        best.append((song, sim))\n",
        "    srtd = sorted(best, key = lambda x: x[1], reverse=True)\n",
        "    for t in srtd[:10]:\n",
        "        print(\"Song: \", t[0], \"  Score: {:.3f}\".format(t[1]))"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 32,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "3G2AdF-2HkBB",
        "outputId": "e53be0d4-54ac-4600-abbf-7c0d22d1ab43"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "<ipython-input-29-081fac839062>:8: RuntimeWarning: invalid value encountered in double_scalars\n",
            "  return (num ) /  (denom )\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Song:  I Love NYC   Score: 0.707\n",
            "Song:  New York   Score: 0.686\n",
            "Song:  King Of New York   Score: 0.517\n",
            "Song:  New York City Cops   Score: 0.512\n",
            "Song:  Stranger Into Starman   Score: 0.333\n",
            "Song:  Cocaine In My Brain   Score: 0.319\n",
            "Song:  Feeling Good   Score: 0.298\n",
            "Song:  The World I Know   Score: 0.239\n",
            "Song:  New Noise   Score: 0.214\n",
            "Song:  Brand New Day   Score: 0.189\n"
          ]
        }
      ],
      "source": [
        "best_songs_for_city(\"New York City\")"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 33,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "EHPdIHRbHkBB",
        "outputId": "6cfe4fdd-cbd8-4498-8309-6c54dbe67e6f"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "<ipython-input-29-081fac839062>:8: RuntimeWarning: invalid value encountered in double_scalars\n",
            "  return (num ) /  (denom )\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Song:  London Bridge   Score: 0.761\n",
            "Song:  London Is The Place For Me   Score: 0.727\n",
            "Song:  London Calling   Score: 0.617\n",
            "Song:  Glamorous Glue   Score: 0.523\n",
            "Song:  London Loves   Score: 0.514\n",
            "Song:  Street Fighting Man   Score: 0.280\n",
            "Song:  Your Embrace   Score: 0.225\n",
            "Song:  The Vanishing   Score: 0.185\n",
            "Song:  Round Here   Score: 0.149\n",
            "Song:  Delaney   Score: 0.143\n"
          ]
        }
      ],
      "source": [
        "best_songs_for_city(\"London\")"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 34,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "FUdyHbHkHkBC",
        "outputId": "3cf5fe2a-658a-4c6f-9ad3-0d2ed4c9d825"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "<ipython-input-29-081fac839062>:8: RuntimeWarning: invalid value encountered in double_scalars\n",
            "  return (num ) /  (denom )\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Song:  Proud To Be Canadian   Score: 0.235\n",
            "Song:  Christian Or Canadian   Score: 0.203\n",
            "Song:  I Love NYC   Score: 0.173\n",
            "Song:  North American Scum   Score: 0.167\n",
            "Song:  North To Alaska   Score: 0.158\n",
            "Song:  New York   Score: 0.154\n",
            "Song:  Shadowplay   Score: 0.151\n",
            "Song:  King Of New York   Score: 0.139\n",
            "Song:  The Ballot or the Bullet   Score: 0.126\n",
            "Song:  New York City Cops   Score: 0.119\n"
          ]
        }
      ],
      "source": [
        "best_songs_for_city(\"Toronto\")"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 35,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "tg2mVO5vHkBC",
        "outputId": "7d256f8d-91cd-4e5a-ec9f-2b5fa3512ae7"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "<ipython-input-29-081fac839062>:8: RuntimeWarning: invalid value encountered in double_scalars\n",
            "  return (num ) /  (denom )\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Song:  Tokyo Witch   Score: 0.280\n",
            "Song:  Panda Bear   Score: 0.163\n",
            "Song:  Never Ending Summer   Score: 0.142\n",
            "Song:  Award Tour   Score: 0.104\n",
            "Song:  Back 4 U   Score: 0.075\n",
            "Song:  Bodhisattva   Score: 0.072\n",
            "Song:  Impossible Germany   Score: 0.063\n",
            "Song:  Harajuku Girls   Score: 0.059\n",
            "Song:  Da Joint   Score: 0.055\n",
            "Song:  Losing My Edge   Score: 0.050\n"
          ]
        }
      ],
      "source": [
        "best_songs_for_city(\"Tokyo\")"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 36,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "NZ5urYHpHkBC",
        "outputId": "d39bd0dc-5358-4d24-894f-e2f22ba40fe6"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "<ipython-input-29-081fac839062>:8: RuntimeWarning: invalid value encountered in double_scalars\n",
            "  return (num ) /  (denom )\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Song:  Indian Girl   Score: 0.158\n",
            "Song:  Citysong   Score: 0.129\n",
            "Song:  The Ballot or the Bullet   Score: 0.124\n",
            "Song:  Fireworks   Score: 0.112\n",
            "Song:  30 Century Man   Score: 0.107\n",
            "Song:  21st Century Digital Boy   Score: 0.088\n",
            "Song:  Twentieth Century Fox   Score: 0.084\n",
            "Song:  Suburban Home   Score: 0.083\n",
            "Song:  The Kids   Score: 0.082\n",
            "Song:  Flux   Score: 0.080\n"
          ]
        }
      ],
      "source": [
        "best_songs_for_city(\"Mumbai\")"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 37,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "xhqGyl5hHkBC",
        "outputId": "67f29405-d1f9-4fa7-9fad-4cb5b070be6d"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "<ipython-input-29-081fac839062>:8: RuntimeWarning: invalid value encountered in double_scalars\n",
            "  return (num ) /  (denom )\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Song:  Island Home   Score: 0.174\n",
            "Song:  Total Life Forever   Score: 0.164\n",
            "Song:  The Ballot or the Bullet   Score: 0.150\n",
            "Song:  Rock Island Line   Score: 0.140\n",
            "Song:  Square Biz   Score: 0.139\n",
            "Song:  Square Dance   Score: 0.126\n",
            "Song:  Jackson Square   Score: 0.125\n",
            "Song:  Island   Score: 0.122\n",
            "Song:  30 Century Man   Score: 0.117\n",
            "Song:  21st Century Life   Score: 0.114\n"
          ]
        }
      ],
      "source": [
        "best_songs_for_city(\"Budapest\")"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 38,
      "metadata": {
        "id": "nCfn7fuBHkBC"
      },
      "outputs": [],
      "source": [
        "with open('wiki_tf_idf.pkl', 'wb') as f:\n",
        "    pickle.dump(vec2, f)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 39,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 177
        },
        "id": "A_EGL-iMHkBD",
        "outputId": "d7f03b84-1569-438d-90f0-9a334f1ca591"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "             title  tag         artist  year  views features  \\\n",
              "27228  Island Home  pop  Christine Anu  1995   1286       {}   \n",
              "\n",
              "                                                  lyrics      id  \\\n",
              "27228  Six years ive been in the city\\nAnd everynight...  854752   \n",
              "\n",
              "                  song_id         emotions  \\\n",
              "27228  TRLWWET128F422729C  [(intense, 50)]   \n",
              "\n",
              "                                             social_tags  \\\n",
              "27228  [(90s, 100), (australian, 100), (chillout, 50)...   \n",
              "\n",
              "                                            tknzd_lyrics  \n",
              "27228  [six, years, ive, been, in, the, city, and, ev...  "
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-febab368-32a6-4ab4-8a46-a4196c7d53fe\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>title</th>\n",
              "      <th>tag</th>\n",
              "      <th>artist</th>\n",
              "      <th>year</th>\n",
              "      <th>views</th>\n",
              "      <th>features</th>\n",
              "      <th>lyrics</th>\n",
              "      <th>id</th>\n",
              "      <th>song_id</th>\n",
              "      <th>emotions</th>\n",
              "      <th>social_tags</th>\n",
              "      <th>tknzd_lyrics</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>27228</th>\n",
              "      <td>Island Home</td>\n",
              "      <td>pop</td>\n",
              "      <td>Christine Anu</td>\n",
              "      <td>1995</td>\n",
              "      <td>1286</td>\n",
              "      <td>{}</td>\n",
              "      <td>Six years ive been in the city\\nAnd everynight...</td>\n",
              "      <td>854752</td>\n",
              "      <td>TRLWWET128F422729C</td>\n",
              "      <td>[(intense, 50)]</td>\n",
              "      <td>[(90s, 100), (australian, 100), (chillout, 50)...</td>\n",
              "      <td>[six, years, ive, been, in, the, city, and, ev...</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-febab368-32a6-4ab4-8a46-a4196c7d53fe')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-febab368-32a6-4ab4-8a46-a4196c7d53fe button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-febab368-32a6-4ab4-8a46-a4196c7d53fe');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 39
        }
      ],
      "source": [
        "big_df.loc[big_df['title']=='Island Home']"
      ]
    }
  ],
  "metadata": {
    "kernelspec": {
      "display_name": "Python 3",
      "language": "python",
      "name": "python3"
    },
    "language_info": {
      "codemirror_mode": {
        "name": "ipython",
        "version": 3
      },
      "file_extension": ".py",
      "mimetype": "text/x-python",
      "name": "python",
      "nbconvert_exporter": "python",
      "pygments_lexer": "ipython3",
      "version": "3.8.5"
    },
    "colab": {
      "provenance": [],
      "include_colab_link": true
    }
  },
  "nbformat": 4,
  "nbformat_minor": 0
}