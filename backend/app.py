import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
from return_songs import find_nonzero_indices

# https://spotipy.readthedocs.io/en/2.22.1/
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials


# ROOT_PATH for linking with all your files.
# Feel free to use a config.py or settings.py with a global export variable
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..", os.curdir))

# These are the DB credentials for your OWN MySQL
# Don't worry about the deployment credentials, those are fixed
# You can use a different DB name if you want to
CREDS = json.load(open('./conf.json', 'r+'))

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

sp = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials(CLIENT_ID, CLIENT_PRIVATE))

# print('hey hey hey heres a playlist', sp.playlist('6Lt5r7BdOM87jphRlYA1zv'))

@app.route("/")
def home():
    return render_template('home.html', title="sample html")


@app.route('/my-link')
def my_link():
    current_url = request.url
    city = current_url[current_url.index('key=') + 4:]
    cityClean = city.replace('_', ' ')
    content = find_nonzero_indices(cityClean)

    content_integrated = {}

    # spotify integration
    # Note: this needs to be HEAVILY refined

    for song in content:
        q = ""

        title = song[0].lower().replace(' ', '%20')
        artist = song[1].lower().replace(' ', '%20')
        year = str(song[2])

        q = "track\:" + title + "%20artist\:" + artist + "%20year\:" + year
        
        result = sp.search(q, limit=1, type='track')
        track = result['tracks']['items']

        key = song[0]

        content_integrated[key] = {'song': '', 'song_link': '', 'artists': [], 'artists_links': [], 'album_art': '', 'album_link': '', 'year': ''}

        if (len(track) > 0) and ((song[0].lower() in track[0]['name'].lower()) or (track[0]['artists'][0]['name'].lower() in song[1].lower())):
            track = track[0]

            # album art, album link, song name, song link, artists, artists links, year

            # get artist(s)
            for i in track['artists']:
                content_integrated[key]['artists'].append(i['name'])
                content_integrated[key]['artists_links'].append(i['uri'])

            # get song name & link
            content_integrated[key]['song'] = song[0]
            content_integrated[key]['song_link'] = track['uri']
            
            # get album art & link
            content_integrated[key]['album_art'] = track['album']['images'][1]['url']
            content_integrated[key]['album_link'] = track['album']['uri']

            # get year
            content_integrated[key]['year'] = year
        # elif (len(track) > 0)
        else:
            # print(len(track))
            # print(track)
            
            content_integrated[key]['artists'].append(song[1])

            content_integrated[key]['song'] = song[0]
            content_integrated[key]['year']

    return render_template('results.html', data=content_integrated)


@app.route("/episodes")
def episodes_search():
    text = request.args.get("title")
    return sql_search(text)

'''
@app.route('/results')
def results():
    return render_template('results.html')
'''


app.run(debug=True)
