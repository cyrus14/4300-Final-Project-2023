import json
import pandas as pd
import os
import shutil
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
import helpers.query as apq
from return_songs import find_nonzero_indices
import plotly.express as px
import pandas as pd
import numpy as np

# https://spotipy.readthedocs.io/en/2.22.1/
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials


### --- SETUP --- ###

# ROOT_PATH for linking with all your files
root_path = os.path.abspath(os.curdir)
os.environ['ROOT_PATH'] = root_path

# These are the DB credentials for your OWN MySQL
# Don't worry about the deployment credentials, those are fixed
# You can use a different DB name if you want to
CREDS = json.load(open(os.environ['ROOT_PATH'] + '/conf.json', 'r+'))

app = Flask(__name__)
CORS(app)

CLIENT_ID = CREDS['spotify_client']
CLIENT_PRIVATE = CREDS['spotify_private']

sp = spotipy.Spotify(
    client_credentials_manager=SpotifyClientCredentials(CLIENT_ID, CLIENT_PRIVATE))


### --- ROUTING --- ###

# home page
@app.route("/")
def home():
    # clears dataviz directory if user is starting a new query
    try:
        shutil.rmtree(os.environ['ROOT_PATH'] + '/static/viz')
    except FileNotFoundError as e:
        pass
    # return home page
    return render_template('home.html', title="sample html")

# results page
@app.route('/results')
def my_link():
    # query tokenization
    current_url = request.url

    urlQuery = current_url[current_url.index('?') + 1:]
    urlQuerySplit = urlQuery.split('&')
    cityRaw = urlQuerySplit[0]
    moodsRaw = urlQuerySplit[1]
    cityClean = cityRaw.replace("key=", "").replace('_', ' ')
    moodsClean = moodsRaw.replace("moods=", "").replace('_', ' ')

    content = apq.top_songs_query(cityClean, query=moodsClean)

    # storing content for recommendation results
    content_integrated = {}

    # storing visualizations for recommendation explainability
    if not os.path.exists("static/viz"):
        os.mkdir("static/viz")

    for item in content:
        year = str(item['year'])

        # query construction
        spq = "track\:" + item['title'] + "%20artist\:" + item['artist']

        # query result
        results = sp.search(spq, limit=1, market='US', type='track')
        track = results['tracks']['items']

        # create entry for song in content_integrated
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
                                   'score': round(item['score'] * 100.0, 2)}

        # if we get a result...
        if (len(track) > 0 and ((item['title'].lower() in track[0]['name'].lower()) or (track[0]['artists'][0]['name'].lower() in item['artist'].lower()))):
            track = track[0]

            # store data in result
            for i in track['artists']:
                content_integrated[key]['artists'].append(i['name'])
                content_integrated[key]['artists_links'].append(i['uri'])
            content_integrated[key]['song'] = item['title']
            content_integrated[key]['song_link'] = track['uri']
            content_integrated[key]['year'] = year
            try:
                content_integrated[key]['album_art'] = track['album']['images'][1]['url']
            except IndexError as e:
                # rarely, we get a song with no album art up on Spotify; this is how we mitigate it
                pass
            content_integrated[key]['album_link'] = track['album']['uri']
            content_integrated[key]['preview_url'] = track['preview_url']
            content_integrated[key]['id'] = track['id']

            # number/data manipulation for visualization
            temp_df = pd.DataFrame.from_dict(item)
            temp_df['score_in_city'] = temp_df['score_in_city'] / \
                np.linalg.norm(temp_df['score_in_city'])
            temp_df['score_in_song'] = temp_df['score_in_song'] / \
                np.linalg.norm(temp_df['score_in_song'])
            temp_df = pd.melt(temp_df, id_vars=['best_words'], value_vars=[
                              'score_in_city', 'score_in_song'])

            # visualizing data
            fig = px.line_polar(
                data_frame=temp_df,
                r='value',
                theta='best_words',
                color='variable',
                color_discrete_sequence=['white', 'magenta'],
                line_close=True,
                template='plotly_dark',
                log_r=True,
                height=500,
                width=500,
                # line_shape="spline",
                title="Term prevalence"
                # range_r=[0, max(temp_df['value'])],
            )
            fig.update_traces(
                fill='toself',
                opacity=0.5
            )
            fig.update_layout(
                font_family = 'DM Sans, sans-serif',
                title_font_family = 'DM Sans, sans-serif',
                font=dict(size=18),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                title=dict(
                    xanchor="center",
                    yanchor="top",
                    x=0.5,
                    y=0.92
                ),
                legend=dict(
                    orientation="h",
                    title="",
                    y=-0.2,
                    x=0.5,
                    yanchor="bottom",
                    xanchor="center",
                    bgcolor="rgba(0,0,0,0)",
                ),
            )
            fig.update_polars(
                # angularaxis_showline=False,
                # angularaxis_showgrid=False,
                angularaxis_linecolor="rgba(220, 220, 220, 0.2)",
                radialaxis_linecolor="rgba(220, 220, 220, 0.2)",
                angularaxis_gridcolor="rgba(220, 220, 220, 0.2)",
                radialaxis_showgrid=False,
                radialaxis_ticks="",
                radialaxis_color="rgba(220, 220, 220, 0.2)",
                radialaxis_showticklabels=False,
                radialaxis_tickcolor="rgba(0,0,0,0)",
                bgcolor="rgba(0,0,0,0)",

            )
            newnames = {"score_in_city": "city score",
                        "score_in_song": "song score"}
            fig.for_each_trace(lambda t: t.update(name=newnames[t.name]))

            # storing image in file
            temp_filename = "static/viz/" + \
                cityClean.replace(' ', '') + \
                str(content_integrated[key]['id']) + ".svg"
            fig.write_image(temp_filename)
        # if we don't get a result...
        else:
            content_integrated.pop(key)
    # return results page
    return render_template('results.html', data=content_integrated, city=cityClean, cityStripped=cityClean.replace(' ', ''), moods=moodsClean.replace(' ', ", "))

#app.run(debug=True)
