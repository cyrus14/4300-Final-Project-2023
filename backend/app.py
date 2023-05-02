import json
import pandas as pd
import os
import shutil
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
import helpers.query as apq
import pickle
from return_songs import find_nonzero_indices
import plotly.express as px
import kaleido 


# # https://spotipy.readthedocs.io/en/2.22.1/
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials


# ROOT_PATH for linking with all your files.
# Feel free to use a config.py or settings.py with a global export variable
root_path = os.path.abspath(os.curdir)
os.environ['ROOT_PATH'] = root_path

# FOR TESTING LOCALLY
# os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..", os.curdir))

# These are the DB credentials for your OWN MySQL
# Don't worry about the deployment credentials, those are fixed
# You can use a different DB name if you want to
CREDS = json.load(open(os.environ['ROOT_PATH'] + '/conf.json', 'r+'))

MYSQL_USER = CREDS['sql_user']
MYSQL_USER_PASSWORD = CREDS['sql_user_pwd']
MYSQL_PORT = CREDS['sql_port']
MYSQL_DATABASE = CREDS['sql_db']

# mysql_engine = MySQLDatabaseHandler(MYSQL_USER,MYSQL_USER_PASSWORD,MYSQL_PORT,MYSQL_DATABASE)

# # Path to init.sql file. This file can be replaced with your own file for testing on localhost, but do NOT move the init.sql file
# mysql_engine.load_file_into_db()

app = Flask(__name__)
CORS(app)

# Sample search, the LIKE operator in this case is hard-coded,
# but if you decide to use SQLAlchemy ORM framework,
# there's a much better and cleaner way to do this


def sql_search(episode):
    query_sql = f"""SELECT * FROM episodes WHERE LOWER( title ) LIKE '%%{episode.lower()}%%' limit 10"""
    keys = ["id", "title", "descr"]
    data = mysql_engine.query_selector(query_sql)
    return json.dumps([dict(zip(keys, i)) for i in data])

# SPOTIFY CREDENTIAL SETUP


# Lol i dont care rn
CLIENT_ID = CREDS['spotify_client']
CLIENT_PRIVATE = CREDS['spotify_private']

sp = spotipy.Spotify(
    client_credentials_manager=SpotifyClientCredentials(CLIENT_ID, CLIENT_PRIVATE))


@app.route("/")
def home():
    try:
        shutil.rmtree(os.environ['ROOT_PATH'] + '/static/viz')
    except FileNotFoundError as e:
        pass
    return render_template('home.html', title="sample html")


@app.route('/results')
def my_link():
    current_url = request.url

    urlQuery = current_url[current_url.index('?') + 1:]
    urlQuerySplit = urlQuery.split('&');

    cityRaw = urlQuerySplit[0]
    moodsRaw = urlQuerySplit[1]

    cityClean = cityRaw.replace("key=", "").replace('_', ' ')
    moodsClean = moodsRaw.replace("moods=", "").replace('_', ' ')

    content = apq.top_songs_query(cityClean, query=moodsClean)

    content_integrated = {}

    # spotify integration
    # this STILL needs to be heavily refined

    for item in content:
        # title = item['title'].lower().replace(' ', '%20')
        # artist = item['artist'].lower().replace(' ', '%20')
        year = str(item['year'])

        # spotify query construction
        spq = "track\:" + item['title'] + "%20artist\:" + item['artist'] + "%20year\:" + year

        # spotify query result
        results = sp.search(spq, limit=1, type='track')
        track = results['tracks']['items']

        key = item['title']

        content_integrated[key] = {'song': '',
                                   'song_link': '',
                                   'artists': [],
                                   'artists_links': [],
                                   'album_art': '',
                                   'album_link': '',
                                   'year': '',
                                   'preview_url': '',
                                   'id': '',
                                   'sim': round(item['sim'] * 100.0, 2),
                                   'pop': round(item['pop'] * 100.0, 2),
                                   'emot': round(item['emot'] * 100.0, 2),
                                   'score': round(item['score'] * 100.0, 2) }
        

        if (len(track) > 0 and ((item['title'].lower() in track[0]['name'].lower()) or (track[0]['artists'][0]['name'].lower() in item['artist'].lower()))):
            track = track[0]

            for i in track['artists']:
                content_integrated[key]['artists'].append(i['name'])
                content_integrated[key]['artists_links'].append(i['uri'])

            content_integrated[key]['song'] = item['title']
            content_integrated[key]['song_link'] = track['uri']

            content_integrated[key]['year'] = year

            content_integrated[key]['album_art'] = track['album']['images'][1]['url']
            content_integrated[key]['album_link'] = track['album']['uri']
            
            content_integrated[key]['preview_url'] = track['preview_url']

            content_integrated[key]['id'] = track['id']
        else:
            # print(len(track))
            # print(track)

            content_integrated[key]['artists'].append(item['artist'])

            content_integrated[key]['song'] = item['title']
            content_integrated[key]['year'] = year
            content_integrated[key]['id'] = track['id']

    return render_template('results.html', data=content_integrated, city=cityClean, moods=moodsClean.replace(' ', ", "))

@app.route('/test')
def svg_test():
    current_url = request.url

    urlQuery = current_url[current_url.index('?') + 1:]
    urlQuerySplit = urlQuery.split('&')

    cityRaw = urlQuerySplit[0]
    moodsRaw = urlQuerySplit[1]

    cityClean = cityRaw.replace("key=", "").replace('_', ' ')
    moodsClean = moodsRaw.replace("moods=", "").replace('_', ' ')

    content = apq.top_songs_query(cityClean, query=moodsClean)

    if not os.path.exists("static/viz"):
        os.mkdir("static/viz")

    for item in content:
        temp_df = pd.DataFrame.from_dict(item)
        temp_df['score_in_city'] = temp_df['score_in_city'] * 10.
        temp_df = pd.melt(temp_df, id_vars=['best_words'], value_vars=['score_in_city', 'score_in_song'])

        fig = px.line_polar(
            data_frame=temp_df,
            r='value',
            theta='best_words',
            color='variable',
            color_discrete_sequence=['magenta', 'dodgerblue'],
            line_close=True,
            template='plotly_dark',
            log_r=False,
            range_r=[0, max(temp_df['value'])],
        )

        fig.update_traces(fill='toself')

        temp_filename = "static/viz/" + cityClean.replace(' ', '') + str(item['id']) + ".svg"

        fig.write_image(temp_filename)


    return render_template('test.html', data=content, city=cityClean.replace(' ', ''), moods=moodsClean.replace(' ', ","))


# app.run(debug=False)
